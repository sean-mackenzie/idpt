"""Pydantic models for job settings - replaces Excel input."""

from typing import Optional
from pydantic import BaseModel, Field


class CalibrationInputConfig(BaseModel):
    """Calibration image input configuration."""

    image_base_string: str = Field(
        default="calib_",
        description="Base string prefix for calibration image filenames",
    )
    image_file_type: str = Field(
        default="tif",
        description="Image file extension (without dot)",
    )
    z_step_size: float = Field(
        default=1.0,
        description="Z-step size between calibration images in microns",
    )
    baseline_image: str = Field(
        default="calib_50.tif",
        description="Filename of baseline/reference image",
    )
    hard_baseline: bool = Field(
        default=True,
        description="Use hard baseline for particle identification",
    )
    image_subset: Optional[list[int]] = Field(
        default=None,
        description="Optional subset of image indices to use",
    )


class TestInputConfig(BaseModel):
    """Test image input configuration."""

    image_base_string: str = Field(
        default="test_",
        description="Base string prefix for test image filenames",
    )
    image_file_type: str = Field(
        default="tif",
        description="Image file extension (without dot)",
    )
    baseline_image: str = Field(
        default="test_39.tif",
        description="Filename of baseline image for particle identification",
    )
    hard_baseline: bool = Field(
        default=True,
        description="Use hard baseline for particle identification",
    )
    image_subset: Optional[list[int]] = Field(
        default=None,
        description="Optional subset of image indices to use",
    )


class ProcessingConfig(BaseModel):
    """Image processing configuration."""

    min_particle_area: int = Field(
        default=2,
        ge=1,
        description="Minimum particle area in pixels",
    )
    max_particle_area: int = Field(
        default=1000,
        ge=1,
        description="Maximum particle area in pixels",
    )
    template_padding: int = Field(
        default=17,
        ge=1,
        description="Padding around particle templates in pixels",
    )
    same_id_threshold: float = Field(
        default=3.0,
        ge=0,
        description="Maximum distance threshold for same particle identification",
    )
    stacks_use_raw: bool = Field(
        default=True,
        description="Use raw images for stacks",
    )
    threshold_method: str = Field(
        default="manual",
        description="Thresholding method (manual, otsu, etc.)",
    )
    threshold_value: float = Field(
        default=1200,
        description="Threshold value for manual thresholding",
    )
    cropping_pad: int = Field(
        default=5,
        ge=0,
        description="Padding for image cropping",
    )
    xy_displacement: Optional[list[list[int]]] = Field(
        default=None,
        description="XY displacement correction [[dx, dy], ...]",
    )


class TestProcessingConfig(ProcessingConfig):
    """Test-specific processing configuration with different defaults."""

    template_padding: int = Field(
        default=14,
        ge=1,
        description="Padding around particle templates in pixels",
    )
    xy_displacement: Optional[list[list[int]]] = Field(
        default=[[-2, -6]],
        description="XY displacement correction [[dx, dy], ...]",
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    save_id_string: str = Field(
        default="idpt",
        description="Identifier string for saved output files",
    )
    save_plots: bool = Field(
        default=True,
        description="Save visualization plots",
    )


class JobSettings(BaseModel):
    """Complete job settings combining all configurations."""

    name: str = Field(
        default="IDPT Job",
        description="Human-readable job name",
    )
    calibration_input: CalibrationInputConfig = Field(
        default_factory=CalibrationInputConfig,
        description="Calibration image input settings",
    )
    test_input: TestInputConfig = Field(
        default_factory=TestInputConfig,
        description="Test image input settings",
    )
    calibration_processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig,
        description="Calibration image processing settings",
    )
    test_processing: TestProcessingConfig = Field(
        default_factory=TestProcessingConfig,
        description="Test image processing settings",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output settings",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Example IDPT Job",
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
                "calibration_processing": {
                    "min_particle_area": 2,
                    "max_particle_area": 1000,
                    "template_padding": 17,
                    "threshold_method": "manual",
                    "threshold_value": 1200,
                },
                "test_processing": {
                    "template_padding": 14,
                    "xy_displacement": [[-2, -6]],
                },
                "output": {
                    "save_id_string": "idpt_example",
                    "save_plots": True,
                },
            }
        }
