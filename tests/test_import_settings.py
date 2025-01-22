import pandas as pd

def read_settings_to_dict(filepath):

    settings_int = ['calibration_template_padding', 'test_template_padding',
                    'share_min_particle_area', 'share_max_particle_area']
    settings_str = ['calibration_image_path', 'test_image_path',
                    'calibration_image_base_string', 'test_image_base_string',
                    'calibration_baseline_image', 'test_baseline_image',
                    'share_save_path', 'share_save_id']
    settings_float = ['calibration_z_step_size', 'share_same_id_threshold']
    settings_bool = ['share_save_plots']
    settings_eval = ['calibration_image_subset', 'test_image_subset',
                     'test_xy_displacement', 'share_cropping', 'share_thresholding']

    df = pd.read_excel(filepath, index_col=0)
    ks = df.index.values.tolist()
    vs = df.v.values.tolist()

    dict_settings = {}
    for k, v in zip(ks, vs):
        if k in settings_int:
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

if __name__ == "__main__":
    FILEPATH = '/Users/mackenzie/Desktop/idpt_example_settings.xlsx'
    DICT_SETTINGS = read_settings_to_dict(filepath=FILEPATH)
    print(DICT_SETTINGS)