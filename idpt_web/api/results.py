"""Results download endpoints."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_db
from ..models.job import ResultsInfo
from ..services.job_service import JobService
from ..services.file_service import FileService

router = APIRouter(prefix="/jobs", tags=["results"])


@router.get("/{job_id}/results", response_model=ResultsInfo)
async def get_results_info(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get information about available results for a job."""
    job_service = JobService(db)

    # Verify job exists
    job = await job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get result files
    result_files = FileService.list_result_files(job_id)

    return ResultsInfo(
        job_id=job_id,
        has_results=len(result_files) > 0,
        files=result_files,
    )


@router.get("/{job_id}/results/download")
async def download_results(
    job_id: str,
    filename: str = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Download result files.

    If filename is specified, download that specific file.
    Otherwise, return the first xlsx file found.
    """
    job_service = JobService(db)

    # Verify job exists
    job = await job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check job is completed
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job.status})",
        )

    results_path = FileService.get_results_path(job_id)
    result_files = FileService.list_result_files(job_id)

    if not result_files:
        raise HTTPException(status_code=404, detail="No results available")

    # Determine which file to download
    if filename:
        if filename not in result_files:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        file_to_download = filename
    else:
        # Default to first xlsx file or first available file
        xlsx_files = [f for f in result_files if f.endswith(".xlsx")]
        file_to_download = xlsx_files[0] if xlsx_files else result_files[0]

    file_path = results_path / file_to_download

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".json": "application/json",
        ".png": "image/png",
        ".csv": "text/csv",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=file_path,
        filename=file_to_download,
        media_type=media_type,
    )
