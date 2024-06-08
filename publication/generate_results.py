# publication/generate_results.py

from publication import settings
from idpt.IdptProcess import IdptProcess


if __name__ == '__main__':
    dataset = '11.06.21_z-micrometer-v2'

    calib_settings = settings.datasets(dataset=dataset, collection_type='calibration').unpack()
    test_settings = settings.datasets(dataset=dataset, collection_type='test').unpack()

    IdptProcess(calib_settings, test_settings).process()