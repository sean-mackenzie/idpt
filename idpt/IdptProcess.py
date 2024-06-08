from .IdptCalibrationModel import IdptCalibrationModel
from .IdptCalibrationStack import IdptCalibrationStack
from .IdptImageCollection import IdptImageCollection


class IdptProcess(object):
    def __init__(self, calib_settings, test_settings):
        self.calib_settings = calib_settings
        self.test_settings = test_settings

    def process(self):
        # calibration collection
        calib_col = IdptImageCollection(self.calib_settings.inputs.image_collection_type,
                                        self.calib_settings.inputs.image_path,
                                        self.calib_settings.inputs.image_file_type,
                                        self.calib_settings.inputs.image_base_string,
                                        self.calib_settings.inputs.calibration_z_step_size,
                                        self.calib_settings.inputs.image_subset,
                                        self.calib_settings.inputs.baseline_image,
                                        self.calib_settings.inputs.hard_baseline,
                                        self.calib_settings.processing.cropping,
                                        self.calib_settings.processing.preprocessing,
                                        self.calib_settings.processing.thresholding,
                                        self.calib_settings.processing.background_subtraction,
                                        self.calib_settings.processing.min_particle_area,
                                        self.calib_settings.processing.max_particle_area,
                                        self.calib_settings.processing.template_padding,
                                        self.calib_settings.processing.same_id_threshold,
                                        self.calib_settings.processing.stacks_use_raw,
                                        self.calib_settings.processing.xy_displacement,
                                        )

        # method for converting filenames to z-coordinates
        name_to_z = {}
        for image in calib_col.images.values():
            name_to_z.update({image.filename: float(
                image.filename.split(self.calib_settings.inputs.image_base_string)[-1].split('.tif')[0])})

        # calibration model
        calib_set = calib_col.create_calibration(name_to_z=name_to_z,
                                                 template_padding=self.calib_settings.processing.template_padding,
                                                 )

        # test collection
        particle_id_image = self.test_settings.inputs.baseline_image
        self.test_settings.inputs.baseline_image = calib_set

        test_col = IdptImageCollection(self.test_settings.inputs.image_collection_type,
                                       self.test_settings.inputs.image_path,
                                       self.test_settings.inputs.image_file_type,
                                       self.test_settings.inputs.image_base_string,
                                       self.test_settings.inputs.calibration_z_step_size,
                                       self.test_settings.inputs.image_subset,
                                       self.test_settings.inputs.baseline_image,
                                       self.test_settings.inputs.hard_baseline,
                                       self.test_settings.processing.cropping,
                                       self.test_settings.processing.preprocessing,
                                       self.test_settings.processing.thresholding,
                                       self.test_settings.processing.background_subtraction,
                                       self.test_settings.processing.min_particle_area,
                                       self.test_settings.processing.max_particle_area,
                                       self.test_settings.processing.template_padding,
                                       self.test_settings.processing.same_id_threshold,
                                       self.test_settings.processing.stacks_use_raw,
                                       self.test_settings.processing.xy_displacement,
                                       particle_id_image=particle_id_image,
                                       )

        test_col.infer_z(calib_set).skncorr(use_stack=None)

        j = 1