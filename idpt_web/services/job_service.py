"""Job CRUD operations service."""

import json
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models.job import JobStatus, JobResponse, JobListResponse
from ..models.settings import JobSettings
from ..schemas.job import Job


class JobService:
    """Service for job CRUD operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_job(self, job_settings: JobSettings) -> JobResponse:
        """Create a new job."""
        job_id = str(uuid.uuid4())

        # Create job directories
        job_upload_dir = settings.uploads_dir / job_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)
        (job_upload_dir / "calibration").mkdir(exist_ok=True)
        (job_upload_dir / "test").mkdir(exist_ok=True)

        job_results_dir = settings.results_dir / job_id
        job_results_dir.mkdir(parents=True, exist_ok=True)

        # Create database record
        job = Job(
            id=job_id,
            name=job_settings.name,
            status=JobStatus.PENDING,
            progress=0.0,
            settings_json=job_settings.model_dump_json(),
            has_calibration_images=False,
            has_test_images=False,
            created_at=datetime.utcnow(),
        )

        self.db.add(job)
        await self.db.flush()

        return self._job_to_response(job)

    async def get_job(self, job_id: str) -> Optional[JobResponse]:
        """Get a job by ID."""
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            return None
        return self._job_to_response(job)

    async def list_jobs(
        self,
        page: int = 1,
        page_size: int = 10,
    ) -> JobListResponse:
        """List jobs with pagination."""
        # Get total count
        count_result = await self.db.execute(select(func.count(Job.id)))
        total = count_result.scalar()

        # Get paginated jobs
        offset = (page - 1) * page_size
        result = await self.db.execute(
            select(Job).order_by(Job.created_at.desc()).offset(offset).limit(page_size)
        )
        jobs = result.scalars().all()

        return JobListResponse(
            jobs=[self._job_to_response(job) for job in jobs],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job and its files."""
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            return False

        # Delete job directories
        import shutil

        job_upload_dir = settings.uploads_dir / job_id
        job_results_dir = settings.results_dir / job_id

        if job_upload_dir.exists():
            shutil.rmtree(job_upload_dir)
        if job_results_dir.exists():
            shutil.rmtree(job_results_dir)

        # Delete database record
        await self.db.delete(job)
        return True

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: float = None,
        error: str = None,
    ) -> Optional[JobResponse]:
        """Update job status."""
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            return None

        job.status = status
        if progress is not None:
            job.progress = progress
        if error is not None:
            job.error = error
        if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
            job.completed_at = datetime.utcnow()

        await self.db.flush()
        return self._job_to_response(job)

    async def update_job_images(
        self,
        job_id: str,
        has_calibration: bool = None,
        has_test: bool = None,
    ) -> Optional[JobResponse]:
        """Update job image upload status."""
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            return None

        if has_calibration is not None:
            job.has_calibration_images = has_calibration
        if has_test is not None:
            job.has_test_images = has_test

        await self.db.flush()
        return self._job_to_response(job)

    def _job_to_response(self, job: Job) -> JobResponse:
        """Convert database job to response model."""
        settings_dict = json.loads(job.settings_json)
        return JobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            progress=job.progress,
            has_calibration_images=job.has_calibration_images,
            has_test_images=job.has_test_images,
            created_at=job.created_at,
            completed_at=job.completed_at,
            error=job.error,
            settings=JobSettings(**settings_dict),
        )
