"""Tests for IDPT Web API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "idpt-web"


@pytest.mark.asyncio
async def test_create_job(client: AsyncClient):
    """Test job creation."""
    settings = {
        "name": "Test Job",
        "calibration_input": {
            "image_base_string": "calib_",
            "image_file_type": "tif",
            "z_step_size": 1.0,
            "baseline_image": "calib_50.tif",
        },
        "test_input": {
            "image_base_string": "test_",
            "image_file_type": "tif",
            "baseline_image": "test_39.tif",
        },
    }

    response = await client.post("/api/v1/jobs", json={"settings": settings})
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Job"
    assert data["status"] == "pending"
    assert data["id"] is not None


@pytest.mark.asyncio
async def test_get_job(client: AsyncClient):
    """Test getting job by ID."""
    # Create a job first
    settings = {"name": "Get Test Job"}
    create_response = await client.post("/api/v1/jobs", json={"settings": settings})
    job_id = create_response.json()["id"]

    # Get the job
    response = await client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id
    assert data["name"] == "Get Test Job"


@pytest.mark.asyncio
async def test_get_job_not_found(client: AsyncClient):
    """Test getting non-existent job."""
    response = await client.get("/api/v1/jobs/non-existent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_jobs(client: AsyncClient):
    """Test listing jobs."""
    # Create a few jobs
    for i in range(3):
        await client.post("/api/v1/jobs", json={"settings": {"name": f"Job {i}"}})

    response = await client.get("/api/v1/jobs")
    assert response.status_code == 200
    data = response.json()
    assert len(data["jobs"]) >= 3
    assert data["page"] == 1


@pytest.mark.asyncio
async def test_delete_job(client: AsyncClient):
    """Test deleting a job."""
    # Create a job
    create_response = await client.post(
        "/api/v1/jobs", json={"settings": {"name": "Delete Test"}}
    )
    job_id = create_response.json()["id"]

    # Delete the job
    delete_response = await client.delete(f"/api/v1/jobs/{job_id}")
    assert delete_response.status_code == 204

    # Verify it's gone
    get_response = await client.get(f"/api/v1/jobs/{job_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_start_processing_without_images(client: AsyncClient):
    """Test that starting processing without images fails."""
    # Create a job
    create_response = await client.post(
        "/api/v1/jobs", json={"settings": {"name": "No Images Job"}}
    )
    job_id = create_response.json()["id"]

    # Try to start processing
    response = await client.post(f"/api/v1/jobs/{job_id}/start")
    assert response.status_code == 400
    assert "Calibration images not uploaded" in response.json()["detail"]
