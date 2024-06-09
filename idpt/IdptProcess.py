from .IdptCalibrationModel import IdptCalibrationModel
from .IdptCalibrationStack import IdptCalibrationStack
from .IdptImageCollection import IdptImageCollection
from os.path import join

class IdptProcess(object):
    def __init__(self, calib_settings, test_settings, name_to_z=None, calib_col=None, calib_set=None, test_col=None):
        self.calib_settings = calib_settings
        self.test_settings = test_settings

        self.name_to_z = name_to_z
        self.calib_col = calib_col
        self.calib_set = calib_set
        self.test_col = test_col

    def process(self):
        # calibration collection
        self.calib_col = IdptImageCollection(self.calib_settings.inputs.image_collection_type,
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
        if self.name_to_z is None:
            name_to_z = {}
            for image in self.calib_col.images.values():
                name_to_z.update({image.filename: float(
                    image.filename.split(self.calib_settings.inputs.image_base_string)[-1].split('.tif')[0])})
            self.name_to_z = name_to_z

        # calibration model
        self.calib_set = self.calib_col.create_calibration(name_to_z=self.name_to_z)

        # test collection
        particle_id_image = self.test_settings.inputs.baseline_image
        self.test_settings.inputs.baseline_image = self.calib_set

        self.test_col = IdptImageCollection(self.test_settings.inputs.image_collection_type,
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

        self.test_col.infer_z(self.calib_set).sknccorr(use_stack=None)

        coords = self.test_col.package_particle_positions()
        coords.to_excel(join(self.test_settings.outputs.results_path, self.test_settings.outputs.save_id_string +
                             '_test-coords.xlsx'), index=False)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        for pid in coords['id'].unique():
            df = coords[coords['id'] == pid]
            ax.plot(df['frame'], df['z_sub'], '-o', linewidth=0.5, ms=2)
        ax.set_xlabel('frame')
        ax.set_ylabel('z_sub')
        plt.tight_layout()
        plt.savefig(join(self.test_settings.outputs.results_path, self.test_settings.outputs.save_id_string +
                         '_z-trajectories.png'), dpi=300)
        plt.close()