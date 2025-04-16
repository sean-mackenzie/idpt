# publication/generate_results.py

from publication import settings
from idpt.IdptProcess import IdptProcess


if __name__ == '__main__':
    dataset = '11.06.21_z-micrometer-v2'

    # You can import IDPT image analysis settings via two methods:
    # 1. Read from a native settings file
    """
    calib_settings = settings.datasets(dataset=dataset).unpack(collection_type='calibration')
    test_settings = settings.datasets(dataset=dataset).unpack(collection_type='test')
    """
    # 2. Import from spreadsheet
    FILEPATH = 'idpt_settings.xlsx'

    # Import settings
    calib_settings, test_settings = settings.datasets(dataset=dataset).read_xlsx(filepath=FILEPATH)

    # Process images to perform 3D particle tracking
    IdptProcess(calib_settings, test_settings).process()