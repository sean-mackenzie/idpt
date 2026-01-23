"""Processor service for running IDPT jobs."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpt import IdptProcess

from ..config import settings
from ..models.job import JobStatus
from ..models.settings import JobSettings
from ..schemas.job import Job
from .settings_adapter import SettingsAdapter
from .file_service import FileService

logger = logging.getLogger(__name__)

# Thread pool for running blocking IDPT processing
_executor = ThreadPoolExecutor(max_workers=2)


class ProcessorService:
    """Service for running IDPT processing jobs."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def start_processing(self, job_id: str) -> bool:
        """
        Start processing a job in the background.

        Args:
            job_id: Job ID to process

        Returns:
            True if processing started, False otherwise
        """
        # Get job from database
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()

        if job is None:
            logger.error(f"Job {job_id} not found")
            return False

        if job.status != JobStatus.PENDING:
            logger.warning(f"Job {job_id} is not in pending status")
            return False

        if not job.has_calibration_images or not job.has_test_images:
            logger.warning(f"Job {job_id} is missing images")
            return False

        # Update status to processing
        job.status = JobStatus.PROCESSING
        job.progress = 0.0
        await self.db.flush()
        await self.db.commit()

        # Start processing in background
        asyncio.create_task(self._run_job(job_id))

        return True

    async def _run_job(self, job_id: str) -> None:
        """Run the IDPT processing job."""
        from ..dependencies import async_session

        async with async_session() as db:
            try:
                # Get job settings
                result = await db.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()

                if job is None:
                    logger.error(f"Job {job_id} not found during processing")
                    return

                job_settings = JobSettings(**json.loads(job.settings_json))

                # Get paths
                calibration_path = FileService.get_upload_path(job_id, "calibration")
                test_path = FileService.get_upload_path(job_id, "test")
                results_path = FileService.get_results_path(job_id)

                # Build IDPT settings
                calib_settings, test_settings = SettingsAdapter.build_settings(
                    job_settings, calibration_path, test_path, results_path
                )

                # Update progress
                job.progress = 10.0
                await db.commit()

                # Run processing in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    _executor,
                    self._run_idpt_process,
                    calib_settings,
                    test_settings,
                )

                # Update job to completed
                result = await db.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = JobStatus.COMPLETED
                    job.progress = 100.0
                    from datetime import datetime
                    job.completed_at = datetime.utcnow()
                    await db.commit()

                logger.info(f"Job {job_id} completed successfully")

            except Exception as e:
                logger.exception(f"Job {job_id} failed: {e}")
                # Update job to failed
                try:
                    result = await db.execute(select(Job).where(Job.id == job_id))
                    job = result.scalar_one_or_none()
                    if job:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        from datetime import datetime
                        job.completed_at = datetime.utcnow()
                        await db.commit()
                except Exception as db_error:
                    logger.exception(f"Failed to update job status: {db_error}")

    @staticmethod
    def _run_idpt_process(calib_settings, test_settings) -> None:
        """Run IDPT processing (blocking call, run in thread pool)."""
        processor = IdptProcess(calib_settings, test_settings)
        processor.process()
