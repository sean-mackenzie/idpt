"""SQLAlchemy ORM model for jobs."""

from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, Boolean, Enum
from sqlalchemy.orm import declarative_base

from ..models.job import JobStatus

Base = declarative_base()


class Job(Base):
    """Job database model."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    status = Column(
        Enum(JobStatus),
        default=JobStatus.PENDING,
        nullable=False,
    )
    progress = Column(Float, default=0.0, nullable=False)
    settings_json = Column(Text, nullable=False)  # JSON serialized settings
    has_calibration_images = Column(Boolean, default=False, nullable=False)
    has_test_images = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Job(id={self.id}, name={self.name}, status={self.status})>"
