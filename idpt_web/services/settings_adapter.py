"""Adapter to convert Pydantic models to IdptSetup objects."""

from pathlib import Path

from idpt import IdptSetup

from ..models.settings import JobSettings


class SettingsAdapter:
    """Converts web settings to IdptSetup objects."""

    @staticmethod
    def build_settings(
        job_settings: JobSettings,
        calibration_path: Path,
        test_path: Path,
        results_path: Path,
    ) -> tuple:
        """
        Convert Pydantic JobSettings to IdptSetup calibration and test settings.

        Args:
            job_settings: Web interface settings
            calibration_path: Path to calibration images
            test_path: Path to test images
            results_path: Path for results output

        Returns:
            Tuple of (calib_settings, test_settings)
        """
        calib_settings = SettingsAdapter._build_calibration_settings(
            job_settings, calibration_path, results_path
        )
        test_settings = SettingsAdapter._build_test_settings(
            job_settings, test_path, results_path
        )
        return calib_settings, test_settings

    @staticmethod
    def _build_calibration_settings(
        job_settings: JobSettings,
        calibration_path: Path,
        results_path: Path,
    ):
        """Build calibration IdptSetup."""
        calib_input = job_settings.calibration_input
        calib_proc = job_settings.calibration_processing
        output = job_settings.output

        # Build inputs
        inputs = IdptSetup.inputs(
            dataset="web_job",
            image_collection_type="calibration",
            image_path=str(calibration_path),
            image_file_type=calib_input.image_file_type,
            image_base_string=calib_input.image_base_string,
            calibration_z_step_size=calib_input.z_step_size,
            baseline_image=calib_input.baseline_image,
            hard_baseline=calib_input.hard_baseline,
            image_subset=calib_input.image_subset,
        )

        # Build outputs
        outputs = IdptSetup.outputs(
            results_path=str(results_path),
            save_id_string=output.save_id_string,
            save_plots=output.save_plots,
        )

        # Build processing
        thresholding = {calib_proc.threshold_method: [calib_proc.threshold_value]}
        cropping = {"pad": calib_proc.cropping_pad} if calib_proc.cropping_pad > 0 else None

        processing = IdptSetup.processing(
            min_particle_area=calib_proc.min_particle_area,
            max_particle_area=calib_proc.max_particle_area,
            template_padding=calib_proc.template_padding,
            same_id_threshold=calib_proc.same_id_threshold,
            stacks_use_raw=calib_proc.stacks_use_raw,
            thresholding=thresholding,
            cropping=cropping,
            background_subtraction=None,
            preprocessing=None,
            xy_displacement=calib_proc.xy_displacement,
        )

        return IdptSetup.IdptSetup(
            inputs=inputs,
            outputs=outputs,
            processing=processing,
            z_assessment=None,
            optics=None,
        )

    @staticmethod
    def _build_test_settings(
        job_settings: JobSettings,
        test_path: Path,
        results_path: Path,
    ):
        """Build test IdptSetup."""
        test_input = job_settings.test_input
        test_proc = job_settings.test_processing
        output = job_settings.output

        # Build inputs
        inputs = IdptSetup.inputs(
            dataset="web_job",
            image_collection_type="test",
            image_path=str(test_path),
            image_file_type=test_input.image_file_type,
            image_base_string=test_input.image_base_string,
            calibration_z_step_size=None,
            baseline_image=test_input.baseline_image,
            hard_baseline=test_input.hard_baseline,
            image_subset=test_input.image_subset,
        )

        # Build outputs
        outputs = IdptSetup.outputs(
            results_path=str(results_path),
            save_id_string=output.save_id_string,
            save_plots=output.save_plots,
        )

        # Build processing
        thresholding = {test_proc.threshold_method: [test_proc.threshold_value]}
        cropping = {"pad": test_proc.cropping_pad} if test_proc.cropping_pad > 0 else None

        processing = IdptSetup.processing(
            min_particle_area=test_proc.min_particle_area,
            max_particle_area=test_proc.max_particle_area,
            template_padding=test_proc.template_padding,
            same_id_threshold=test_proc.same_id_threshold,
            stacks_use_raw=test_proc.stacks_use_raw,
            thresholding=thresholding,
            cropping=cropping,
            background_subtraction=None,
            preprocessing=None,
            xy_displacement=test_proc.xy_displacement,
        )

        # Build z_assessment
        z_assessment = IdptSetup.z_assessment(infer_method="sknccorr")

        return IdptSetup.IdptSetup(
            inputs=inputs,
            outputs=outputs,
            processing=processing,
            z_assessment=z_assessment,
            optics=None,
        )
