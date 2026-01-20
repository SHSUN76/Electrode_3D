"""
Pytest configuration and fixtures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_2d_image():
    """Create sample 2D grayscale image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (256, 256), dtype=np.uint8)


@pytest.fixture
def sample_3d_volume():
    """Create sample 3D volume."""
    np.random.seed(42)
    return np.random.randint(0, 3, (64, 64, 64), dtype=np.uint8)


@pytest.fixture
def sample_sphere_volume():
    """Create 3D volume with a sphere."""
    size = 64
    x, y, z = np.ogrid[:size, :size, :size]
    center = size // 2
    radius = size // 4
    sphere = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2 <= radius ** 2
    return sphere.astype(np.uint8)


@pytest.fixture
def sample_electrode_structure():
    """Create sample electrode microstructure with multiple phases."""
    size = 64
    volume = np.zeros((size, size, size), dtype=np.uint8)

    np.random.seed(42)

    # Add random active material particles
    for _ in range(50):
        cx, cy, cz = np.random.randint(5, size - 5, 3)
        r = np.random.randint(3, 8)

        x, y, z = np.ogrid[:size, :size, :size]
        mask = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r ** 2
        volume[mask] = 1

    # Add binder phase around particles
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(volume == 1, iterations=2)
    binder_mask = dilated & (volume == 0)
    volume[binder_mask] = 2

    return volume


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    from electrode_generator.config import Config
    return Config()
