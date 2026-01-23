"""File upload handling service."""

import aiofiles
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from ..config import settings


class FileService:
    """Service for file upload operations."""

    @staticmethod
    async def save_upload(
        job_id: str,
        upload_type: Literal["calibration", "test"],
        files: list[UploadFile],
    ) -> int:
        """
        Save uploaded files to job directory.

        Args:
            job_id: Job ID
            upload_type: Type of upload (calibration or test)
            files: List of uploaded files

        Returns:
            Number of files saved
        """
        upload_dir = settings.uploads_dir / job_id / upload_type
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for file in files:
            if file.filename:
                file_path = upload_dir / file.filename
                async with aiofiles.open(file_path, "wb") as f:
                    content = await file.read()
                    await f.write(content)
                saved_count += 1

        return saved_count

    @staticmethod
    def get_upload_path(
        job_id: str,
        upload_type: Literal["calibration", "test"],
    ) -> Path:
        """Get the upload directory path for a job."""
        return settings.uploads_dir / job_id / upload_type

    @staticmethod
    def get_results_path(job_id: str) -> Path:
        """Get the results directory path for a job."""
        return settings.results_dir / job_id

    @staticmethod
    def list_uploaded_files(
        job_id: str,
        upload_type: Literal["calibration", "test"],
    ) -> list[str]:
        """List uploaded files for a job."""
        upload_dir = settings.uploads_dir / job_id / upload_type
        if not upload_dir.exists():
            return []
        return [f.name for f in upload_dir.iterdir() if f.is_file()]

    @staticmethod
    def list_result_files(job_id: str) -> list[str]:
        """List result files for a job."""
        results_dir = settings.results_dir / job_id
        if not results_dir.exists():
            return []
        return [f.name for f in results_dir.iterdir() if f.is_file()]

    @staticmethod
    def has_images(
        job_id: str,
        upload_type: Literal["calibration", "test"],
    ) -> bool:
        """Check if a job has uploaded images of the specified type."""
        upload_dir = settings.uploads_dir / job_id / upload_type
        if not upload_dir.exists():
            return False
        return any(upload_dir.iterdir())
