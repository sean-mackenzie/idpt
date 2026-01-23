"""Test configuration for IDPT Web tests."""

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from idpt_web.main import app
from idpt_web.schemas import Base
from idpt_web.dependencies import get_db
from idpt_web import config


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    (storage_dir / "uploads").mkdir()
    (storage_dir / "results").mkdir()
    return storage_dir


@pytest_asyncio.fixture
async def test_db(temp_storage):
    """Create test database."""
    db_path = temp_storage / "test.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    engine = create_async_engine(db_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(test_db, temp_storage, monkeypatch):
    """Create test client with mocked dependencies."""
    # Mock storage paths
    monkeypatch.setattr(config.settings, "storage_dir", temp_storage)

    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
