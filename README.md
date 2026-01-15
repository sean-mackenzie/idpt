
# IDPT: Individualized Defocusing Particle Tracking

## Overview
IDPT (Individualized Defocusing Particle Tracking) is a 3D particle tracking software package for dynamic surface profilometry and strain field measurements. The repository provides a practical example and ready-to-use scripts for those interested in using the software. For a complete description of the software and its applications, see the journal publication: https://iopscience.iop.org/article/10.1088/1361-6501/adcceb

## Installation

To get started with IDPT, you'll need to install the necessary dependencies. This project assumes you have Python and pip installed on your system. You can install the required packages using the following command:


```bash
pip install -r requirements.txt
```

## Usage

As a first-pass demonstration, you can run the ready-to-use example script to perform 3D particle tracking on the images provided with this repository. To do so, run the following command: 

```bash
python publication/generate_results.py
```

You can also re-perform the data analysis and recreate the figures presented in the journal publication. To do so, run the following command: 

```bash
python publication/compare_idpt_gdpt.py
```

## Analyze your own images

To use this software package to analyze your own images, there are really only two things you need to do:
1. Create a settings file to instruct IDPT how to perform the analysis: I recommend duplicating, renaming, and editing the idpt_settings.xlsx file.
2. Run IDPT using the settings file as instructions: I recommend editing the script tests/test_external.py. Just replace the  FILEPATH variable with the name and/or path of your new settings file and run the script. That's it. 

## Contributing

Contributions to IDPT are welcome! If you have suggestions or improvements, feel free to fork this repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Sean MacKenzie - *Initial work*

## Acknowledgments

- Huge thanks to Silvan Stettler (EPFL) for developing the initial implementation of GDPyT, which eventually led to IDPT.
- And to all the contributors of various open-source libraries for making projects like this possible.
