
# IDPT: Individualized Defocusing Particle Tracking

## Overview

IDPT (Individualized Defocusing Particle Tracking) is a 3D particle tracking software package for dynamic surface profilometry and strain field measurements.

**Try it now:** [https://idpt-web.fly.dev](https://idpt-web.fly.dev)

For a complete description of the software and its applications, see the journal publication: [Measurement Science and Technology](https://iopscience.iop.org/article/10.1088/1361-6501/adcceb)

## Features

- 3D particle localization using defocused imaging
- Calibration-based depth estimation
- Web interface for easy analysis without coding
- Python API for custom workflows
- Excel-based settings for batch processing

## Quick Start

### Option 1: Web Interface (No Installation Required)

Visit [https://idpt-web.fly.dev](https://idpt-web.fly.dev) to use IDPT directly in your browser:

1. Create a new job with your processing settings
2. Upload calibration images (z-stack with known positions)
3. Upload test images to analyze
4. Start processing and download results

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

## Web Interface

The web interface provides a user-friendly way to run IDPT analysis:

### Running Locally

```bash
# Install web dependencies
pip install -r requirements-web.txt

# Start the server
uvicorn idpt_web.main:app --reload

# Open http://localhost:8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/jobs` | Create job with settings |
| GET | `/api/v1/jobs/{id}` | Get job status |
| POST | `/api/v1/jobs/{id}/upload/calibration` | Upload calibration images |
| POST | `/api/v1/jobs/{id}/upload/test` | Upload test images |
| POST | `/api/v1/jobs/{id}/start` | Start processing |
| GET | `/api/v1/jobs/{id}/results/download` | Download results |

Full API documentation available at `/docs` when running the server.

### Self-Hosting

Deploy your own instance using Docker:

```bash
# Build and run with Docker
docker build -t idpt-web .
docker run -p 8000:8000 idpt-web
```

Or deploy to cloud platforms using the included configuration files:
- `fly.toml` - Fly.io
- `render.yaml` - Render
- `Dockerfile` - Any container platform

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
