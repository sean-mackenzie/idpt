
# IDPT: Individualized Defocusing Particle Tracking

## Overview

IDPT (Individualized Defocusing Particle Tracking) is a 3D particle tracking software package for dynamic surface profilometry and strain field measurements.

For a complete description of the software and its applications, see the journal publication: [Measurement Science and Technology](https://iopscience.iop.org/article/10.1088/1361-6501/adcceb)

## Features

- 3D particle localization using defocused imaging
- Calibration-based depth estimation
- Web interface for easy analysis without coding (under development)
- Python API for custom workflows
- Excel-based settings for batch processing

## Quick Start

### Option 1: Web Interface (Under Development)

A web interface for running IDPT directly in your browser is currently under development and not yet available for use.

### Option 2: Python Package

#### Installation

```bash
# Clone the repository
git clone https://github.com/sean-mackenzie/idpt.git
cd idpt

# Install dependencies
pip install -r requirements.txt
```

#### Run Example

```bash
# Run the example script
python publication/generate_results.py
```

#### Analyze Your Own Images

1. Create a settings file by duplicating and editing `idpt_settings.xlsx`
2. Edit `tests/test_external.py` with your settings file path
3. Run the script

## Project Structure

```
idpt/
├── idpt/                    # Core particle tracking library
│   ├── IdptSetup.py         # Configuration classes
│   ├── IdptProcess.py       # Main processing pipeline
│   └── IdptImageCollection.py
├── idpt_web/                # Web interface (FastAPI)
│   ├── main.py              # Application entry point
│   ├── api/                 # REST API endpoints
│   ├── services/            # Business logic
│   └── static/              # Frontend (HTML/CSS/JS)
├── publication/             # Example data and analysis scripts
│   ├── images/              # Sample calibration & test images
│   ├── settings.py          # Example settings
│   └── generate_results.py  # Demo script
└── tests/                   # Test suite
```

## Web Interface (Under Development)

A web interface for running IDPT analysis is currently under development and not yet available for use.

## Configuration

### Processing Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image_base_string` | Filename prefix | `calib_` / `test_` |
| `z_step_size` | Calibration z-step (μm) | 1.0 |
| `min_particle_area` | Minimum particle size (px) | 2 |
| `max_particle_area` | Maximum particle size (px) | 1000 |
| `template_padding` | Padding around particles (px) | 17 |
| `threshold_value` | Image threshold | 1200 |

See `idpt_settings.xlsx` for a complete list of parameters.

## Reproducing Publication Results

To recreate the figures from the journal publication:

```bash
python publication/compare_idpt_gdpt.py
```

## Contributing

Contributions to IDPT are welcome! If you have suggestions or improvements:

1. Fork this repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use IDPT in your research, please cite:

```bibtex
@article{mackenzie2025idpt,
  title={Individualized defocusing particle tracking for dynamic surface profilometry},
  author={MacKenzie, Sean},
  journal={Measurement Science and Technology},
  year={2025},
  publisher={IOP Publishing}
}
```

## Authors

- Sean MacKenzie - *Initial work*

## Acknowledgments

- Huge thanks to Silvan Stettler (EPFL) for developing the initial implementation of GDPyT, which eventually led to IDPT.
- And to all the contributors of various open-source libraries for making projects like this possible.
