# publication/settings.py

import os
from os.path import join
import pandas as pd
from idpt import IdptSetup


class datasets(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.collection_type = None

    def unpack(self, collection_type):
        if collection_type not in ['calibration', 'test']:
            raise ValueError("Image collection must be in: ['calibration', 'test']")
        self.collection_type = collection_type

        if self.dataset == '11.06.21_z-micrometer-v2':
            if self.collection_type == 'calibration':
                image_path = join('images', 'calibration', '1umSteps')
                image_file_type = 'tif'
                image_base_string = 'calib_'
                calibration_z_step_size = 1
                image_subset = None
                baseline_image = 'calib_50.tif'
                hard_baseline = True

                results_path = join('results', 'calib')
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
                image_path = join('images', 'tests', '3X_5umSteps')
                image_file_type = 'tif'
                image_base_string = 'test_'
                calibration_z_step_size = None
                image_subset = None
                baseline_image = 'test_39.tif'
                hard_baseline = True

                results_path = join('results', 'test')
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


    def read_xlsx(self, filepath):
        settings = read_settings_to_dict(filepath=filepath)

        if not os.path.exists(settings['share_save_path']):
            os.makedirs(settings['share_save_path'])

        inputs_calibration = IdptSetup.inputs(
            dataset=self.dataset,
            image_collection_type='calibration',
            image_path=settings['calibration_image_path'],
            image_file_type=settings['share_image_file_type'],
            image_base_string=settings['calibration_image_base_string'],
            calibration_z_step_size=settings['calibration_z_step_size'],
            image_subset=settings['calibration_image_subset'],
            baseline_image=settings['calibration_baseline_image'],
        )

        processing_calibration = IdptSetup.processing(
            cropping=settings['share_cropping'],
            thresholding=settings['share_thresholding'],
            min_particle_area=settings['share_min_particle_area'],
            max_particle_area=settings['share_max_particle_area'],
            template_padding=settings['calibration_template_padding'],
            same_id_threshold=settings['share_same_id_threshold'],
        )

        inputs_test = IdptSetup.inputs(
            dataset=self.dataset,
            image_collection_type='test',
            image_path=settings['test_image_path'],
            image_file_type=settings['share_image_file_type'],
            image_base_string=settings['test_image_base_string'],
            image_subset=settings['test_image_subset'],
            baseline_image=settings['test_baseline_image'],
        )

        processing_test = IdptSetup.processing(
            cropping=settings['share_cropping'],
            thresholding=settings['share_thresholding'],
            min_particle_area=settings['share_min_particle_area'],
            max_particle_area=settings['share_max_particle_area'],
            template_padding=settings['test_template_padding'],
            same_id_threshold=settings['share_same_id_threshold'],
            xy_displacement=settings['test_xy_displacement'],
        )

        outputs = IdptSetup.outputs(
            results_path=settings['share_save_path'],
            save_id_string=settings['share_save_id'],
            save_plots=settings['share_save_plots'],
        )

        z_assessment = IdptSetup.z_assessment()

        settings_calibration = IdptSetup.IdptSetup(inputs_calibration, outputs, processing_calibration,
                                                   z_assessment=None, optics=None)
        settings_test = IdptSetup.IdptSetup(inputs_test, outputs, processing_test, z_assessment,
                                            optics=None)
        return settings_calibration, settings_test


def read_settings_to_dict(filepath):

    settings_path = ['calibration_image_path', 'test_image_path', 'share_save_path']
    settings_str = ['calibration_image_base_string', 'test_image_base_string',
                    'calibration_baseline_image', 'test_baseline_image',
                    'share_save_id', 'share_image_file_type']
    settings_int = ['calibration_template_padding', 'test_template_padding',
                    'share_min_particle_area', 'share_max_particle_area']
    settings_float = ['calibration_z_step_size', 'share_same_id_threshold']
    settings_bool = ['share_save_plots']
    settings_eval = ['calibration_image_subset', 'test_image_subset',
                     'test_xy_displacement', 'share_cropping', 'share_thresholding']
    # read settings .xlsx
    df = pd.read_excel(filepath, index_col=0)

    # parse root directory
    base_dir = df.loc['base_dir', 'v']
    if base_dir == 'os.getcwd()':
        base_dir = eval(base_dir)

    # parse settings and data types
    ks = df.index.values.tolist()
    vs = df.v.values.tolist()
    dict_settings = {}
    for k, v in zip(ks, vs):
        if k == 'base_dir':
            pass
        elif k in settings_path:
            dict_settings.update({k: join(base_dir, str(v))})
        elif k in settings_int:
            dict_settings.update({k: int(v)})
        elif k in settings_str:
            dict_settings.update({k: str(v)})
        elif k in settings_float:
            dict_settings.update({k: float(v)})
        elif k in settings_bool:
            dict_settings.update({k: bool(v)})
        elif k in settings_eval:
            dict_settings.update({k: eval(v)})
        else:
            raise ValueError("Settings key not understood.")

    return dict_settings