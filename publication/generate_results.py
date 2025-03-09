# publication/generate_results.py

from publication import settings
from idpt.IdptProcess import IdptProcess
from os.path import join


if __name__ == '__main__':
    dataset = '11.06.21_z-micrometer-v2'

    # Import settings from python script
    """
    calib_settings = settings.datasets(dataset=dataset).unpack(collection_type='calibration')
    test_settings = settings.datasets(dataset=dataset).unpack(collection_type='test')
    """
    # Alternatively, import settings from spreadsheet
    FILEPATH = 'idpt_settings_template+3.xlsx'

    FILEPATHS = ['idpt_settings_Lc+3_Lt+3.xlsx', 'idpt_settings_Lc+3_Lt0.xlsx',
                 'idpt_settings_Lc0_Lt-3.xlsx', 'idpt_settings_Lc-3_Lt-3.xlsx']
    for FILEPATH in FILEPATHS:

        calib_settings, test_settings = settings.datasets(dataset=dataset).read_xlsx(filepath=FILEPATH)

        # Process images to perform 3D particle tracking
        IdptProcess(calib_settings, test_settings).process()