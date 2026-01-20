# Electrode 3D Structure Auto-Generator

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete pipeline for generating realistic 3D electrode microstructures from 2D SEM images using SliceGAN deep learning, with COMSOL Multiphysics integration for electrochemical simulations.

## Features

- **SliceGAN Implementation**: 2D to 3D microstructure generation using WGAN-GP
- **Image Preprocessing**: Normalization, segmentation, and data augmentation
- **Voxel-to-Mesh Conversion**: Marching Cubes and TPMS (Triply Periodic Minimal Surfaces) support
- **COMSOL Integration**: Automated electrochemical simulation setup
- **Microstructure Metrics**: Porosity, specific surface area, tortuosity, connectivity analysis
- **Comprehensive Testing**: 68+ unit tests with pytest

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SHSUN76/Electrode_3D.git
cd Electrode_3D

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Generate 3D Electrode Structure

```python
import torch
from models.slicegan.generator import Generator3D

# Initialize generator
generator = Generator3D(nz=512, ngf=64, nc=3, imtype='nphase')
generator.eval()

# Generate 3D structure
with torch.no_grad():
    z = torch.randn(1, 512, 4, 4, 4)
    output = generator(z)

# output shape: (1, 3, 128, 128, 128)
# 3 channels: Active Material, Pore, Binder
labels = output.argmax(dim=1)  # Get phase labels
```

### Convert to Mesh

```python
from postprocessing.mesh_converter import VoxelToMesh

converter = VoxelToMesh(voxel_size=1.0)
vertices, faces, normals, values = converter.convert(labels.numpy()[0])
```

## Project Structure

```
Electrode_3D/
├── configs/                 # YAML configuration files
│   ├── default.yaml
│   ├── training_fast.yaml
│   └── training_hq.yaml
├── comsol/                  # COMSOL Multiphysics integration
│   ├── interface.py         # COMSOL connection
│   └── simulation.py        # Electrochemical simulation
├── data/
│   ├── raw/                 # Raw SEM images
│   ├── processed/           # Preprocessed data
│   └── generated/           # Generated 3D structures
├── docs/
│   ├── PRD.md               # Product Requirements Document
│   ├── ERD.md               # Entity Relationship Diagram
│   └── MVP.md               # MVP Specification
├── electrode_generator/     # Main package
│   ├── config.py            # Configuration classes
│   ├── core.py              # Core pipeline
│   └── cli.py               # Command-line interface
├── models/
│   └── slicegan/            # SliceGAN implementation
│       ├── generator.py     # 3D Generator
│       ├── discriminator.py # 2D Critic (WGAN-GP)
│       └── trainer.py       # Training loop
├── postprocessing/          # Mesh conversion and export
│   ├── mesh_converter.py    # Voxel to mesh
│   └── export.py            # STL/OBJ export
├── preprocessing/           # Image preprocessing
│   ├── image_processor.py   # Core preprocessing
│   └── augmentation.py      # Data augmentation
├── tests/                   # Unit tests
└── utils/
    ├── metrics.py           # Microstructure metrics
    └── visualization.py     # Visualization tools
```

## Configuration

Use YAML configuration files for training:

```yaml
# configs/default.yaml
slicegan:
  nz: 64              # Latent vector dimension
  ngf: 64             # Generator filters
  ndf: 64             # Discriminator filters
  nc: 3               # Number of phases
  batch_size: 8
  lr_g: 0.0001        # Generator learning rate
  lr_d: 0.0004        # Discriminator learning rate
  lambda_gp: 10.0     # Gradient penalty weight
  epochs: 100

preprocessing:
  target_size: [256, 256]
  num_classes: 3
  denoise: true
  augment: true
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Documentation

- [PRD (Product Requirements)](docs/PRD.md)
- [ERD (Entity Relationship Diagram)](docs/ERD.md)
- [MVP Specification](docs/MVP.md)
- [Technical Reference (Korean)](참고자료1.md)

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   2D SEM     │ -> │   SliceGAN   │ -> │   3D Voxel   │
│   Images     │    │   Generator  │    │   Volume     │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                                               v
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   COMSOL     │ <- │   FEM Mesh   │ <- │   Marching   │
│   Simulation │    │   (Gmsh)     │    │   Cubes      │
└──────────────┘    └──────────────┘    └──────────────┘
```

## References

- SliceGAN: [arXiv:1901.10633](https://arxiv.org/abs/1901.10633)
- WGAN-GP: [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
- MicroLib Dataset: [Nature Scientific Data](https://www.nature.com/articles/s41597-022-01744-1)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- SHSUN76 - [GitHub](https://github.com/SHSUN76)

---

*Generated with assistance from Claude Opus 4.5*
