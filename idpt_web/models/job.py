"""Pydantic models for job status and responses."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from .settings import JobSettings


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreate(BaseModel):
    """Request model for creating a new job."""

    settings: JobSettings = Field(
        default_factory=JobSettings,
        description="Job processing settings",
    )


class JobResponse(BaseModel):
    """Response model for job information."""

    id: str = Field(description="Unique job identifier")
    name: str = Field(description="Job name")
    status: JobStatus = Field(description="Current job status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    has_calibration_images: bool = Field(
        default=False,
        description="Whether calibration images have been uploaded",
    )
    has_test_images: bool = Field(
        default=False,
        description="Whether test images have been uploaded",
    )
    created_at: datetime = Field(description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Job completion timestamp",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if job failed",
    )
    settings: JobSettings = Field(description="Job settings")

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Response model for paginated job list."""

    jobs: list[JobResponse] = Field(description="List of jobs")
    total: int = Field(description="Total number of jobs")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of jobs per page")


class FileUploadResponse(BaseModel):
    """Response model for file upload."""

    job_id: str = Field(description="Job ID")
    file_count: int = Field(description="Number of files uploaded")
    upload_type: str = Field(description="Type of upload (calibration/test)")


class ResultsInfo(BaseModel):
    """Information about available results."""

    job_id: str = Field(description="Job ID")
    has_results: bool = Field(description="Whether results are available")
    files: list[str] = Field(default_factory=list, description="List of result files")
