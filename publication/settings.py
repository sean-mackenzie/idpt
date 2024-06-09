# publication/settings.py

import os
from os.path import join
from idpt import IdptSetup


class datasets(object):
    def __init__(self, dataset, collection_type):
        self.dataset = dataset

        if collection_type not in ['calibration', 'test']:
            raise ValueError("Image collection must be in: ['calibration', 'test']")

        self.collection_type = collection_type

    def unpack(self):
        if self.dataset == '11.06.21_z-micrometer-v2':
            base_dir = '/Users/mackenzie/PythonProjects/idpt/publication'
            if self.collection_type == 'calibration':
                image_path = join(base_dir, 'images/calibration/1umSteps')
                image_file_type = 'tif'
                image_base_string = 'calib_'
                calibration_z_step_size = 1
                image_subset = None
                baseline_image = 'calib_50.tif'
                hard_baseline = True

                results_path = join(base_dir, 'results/calib')
                save_id_string = 'calib'
                save_plots = True

                cropping = {'pad': 5}
                background_subtraction = None
                processing_method = None
                processing_filter_type = None
                processing_filter_size = None
                preprocessing = None
                threshold_method = 'manual'
                threshold_modifier = 1200
                thresholding = {threshold_method: [threshold_modifier]}
                min_particle_area = 2
                max_particle_area = 1000
                template_padding = 17
                same_id_threshold = 3
                stacks_use_raw = True
                xy_displacement = None
                infer_method = None

            else:
                image_path = join(base_dir, 'images/tests/3X_5umSteps')
                image_file_type = 'tif'
                image_base_string = 'test_'
                calibration_z_step_size = None
                image_subset = None
                baseline_image = 'test_39.tif'
                hard_baseline = True

                results_path = join(base_dir, 'results/test')
                save_id_string = 'test'
                save_plots = True

                cropping = {'pad': 5}
                background_subtraction = None
                processing_method = None
                processing_filter_type = None
                processing_filter_size = None
                preprocessing = None
                threshold_method = 'manual'
                threshold_modifier = 1200
                thresholding = {threshold_method: [threshold_modifier]}
                min_particle_area = 2
                max_particle_area = 1000
                template_padding = 14
                same_id_threshold = 3
                stacks_use_raw = True
                xy_displacement = [[-2, -6]]
                infer_method = 'sknccorr'

        else:
            raise ValueError("Dataset must be in: ['11.06.21_z-micrometer-v2']")

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        inputs = IdptSetup.inputs(dataset=self.dataset,
                                  image_collection_type=self.collection_type,
                                  image_path=image_path,
                                  image_file_type=image_file_type,
                                  image_base_string=image_base_string,
                                  calibration_z_step_size=calibration_z_step_size,
                                  image_subset=image_subset,
                                  baseline_image=baseline_image,
                                  hard_baseline=hard_baseline,
                                  )

        outputs = IdptSetup.outputs(results_path=results_path,
                                    save_id_string=save_id_string,
                                    save_plots=save_plots,
                                    )

        processing = IdptSetup.processing(cropping=cropping,
                                          preprocessing=preprocessing,
                                          thresholding=thresholding,
                                          background_subtraction=background_subtraction,
                                          min_particle_area=min_particle_area,
                                          max_particle_area=max_particle_area,
                                          template_padding=template_padding,
                                          same_id_threshold=same_id_threshold,
                                          stacks_use_raw=stacks_use_raw,
                                          xy_displacement=xy_displacement,
                                          )

        z_assessment = IdptSetup.z_assessment(infer_method=infer_method)

        if self.collection_type == 'calibration':
            return IdptSetup.IdptSetup(inputs, outputs, processing, z_assessment=None, optics=None)
        else:
            return IdptSetup.IdptSetup(inputs, outputs, processing, z_assessment, optics=None)