"""Job CRUD and file upload endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_db
from ..models.job import JobCreate, JobResponse, JobListResponse, FileUploadResponse
from ..models.settings import JobSettings
from ..services.job_service import JobService
from ..services.file_service import FileService
from ..services.processor import ProcessorService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=201)
async def create_job(
    job_create: JobCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Create a new processing job with settings."""
    job_service = JobService(db)
    return await job_service.create_job(job_create.settings)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=10, ge=1, le=100, description="Items per page"),
):
    """List all jobs with pagination."""
    job_service = JobService(db)
    return await job_service.list_jobs(page=page, page_size=page_size)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get job status and details."""
    job_service = JobService(db)
    job = await job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/{job_id}", status_code=204)
async def delete_job(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Cancel and delete a job."""
    job_service = JobService(db)
    deleted = await job_service.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")


@router.post("/{job_id}/upload/calibration", response_model=FileUploadResponse)
async def upload_calibration_images(
    job_id: str,
    files: list[UploadFile] = File(..., description="Calibration image files"),
    db: AsyncSession = Depends(get_db),
):
    """Upload calibration images for a job."""
    job_service = JobService(db)

    # Verify job exists
    job = await job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Save files
    file_count = await FileService.save_upload(job_id, "calibration", files)

    # Update job status
    has_images = FileService.has_images(job_id, "calibration")
    await job_service.update_job_images(job_id, has_calibration=has_images)

    return FileUploadResponse(
        job_id=job_id,
        file_count=file_count,
        upload_type="calibration",
    )


@router.post("/{job_id}/upload/test", response_model=FileUploadResponse)
async def upload_test_images(
    job_id: str,
    files: list[UploadFile] = File(..., description="Test image files"),
    db: AsyncSession = Depends(get_db),
):
    """Upload test images for a job."""
    job_service = JobService(db)

    # Verify job exists
    job = await job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Save files
    file_count = await FileService.save_upload(job_id, "test", files)

    # Update job status
    has_images = FileService.has_images(job_id, "test")
    await job_service.update_job_images(job_id, has_test=has_images)

    return FileUploadResponse(
        job_id=job_id,
        file_count=file_count,
        upload_type="test",
    )


@router.post("/{job_id}/start", response_model=JobResponse)
async def start_processing(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Start processing a job."""
    job_service = JobService(db)
    processor_service = ProcessorService(db)

    # Verify job exists
    job = await job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if images are uploaded
    if not job.has_calibration_images:
        raise HTTPException(
            status_code=400,
            detail="Calibration images not uploaded",
        )
    if not job.has_test_images:
        raise HTTPException(
            status_code=400,
            detail="Test images not uploaded",
        )

    # Check job status
    if job.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Job is already {job.status}",
        )

    # Start processing
    started = await processor_service.start_processing(job_id)
    if not started:
        raise HTTPException(
            status_code=500,
            detail="Failed to start processing",
        )

    # Return updated job
    return await job_service.get_job(job_id)
