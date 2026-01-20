"""
Tests for metrics calculation module.
"""

import numpy as np
import pytest

from utils.metrics import MicrostructureMetrics, MetricsComparator


class TestMicrostructureMetrics:
    """Tests for MicrostructureMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create metrics calculator instance."""
        return MicrostructureMetrics(voxel_size=1.0)

    @pytest.fixture
    def sample_voxels(self):
        """Create sample voxel data."""
        # Create a simple structure: solid cube in pore matrix
        voxels = np.zeros((32, 32, 32), dtype=np.uint8)
        voxels[8:24, 8:24, 8:24] = 1  # Active material cube
        return voxels

    @pytest.fixture
    def multiphase_voxels(self):
        """Create multi-phase voxel data."""
        voxels = np.zeros((32, 32, 32), dtype=np.uint8)
        voxels[5:15, 5:15, 5:15] = 1  # Active material
        voxels[18:28, 18:28, 18:28] = 2  # Binder
        return voxels

    def test_volume_fraction(self, metrics, sample_voxels):
        """Test volume fraction calculation."""
        vf = metrics.volume_fraction(sample_voxels, phase_id=1)

        # Cube is 16x16x16 = 4096 voxels out of 32x32x32 = 32768
        expected = 4096 / 32768
        assert np.isclose(vf, expected)

    def test_porosity(self, metrics, sample_voxels):
        """Test porosity calculation."""
        porosity = metrics.porosity(sample_voxels, pore_id=0)

        # Porosity is 1 - solid fraction
        expected = 1 - 4096 / 32768
        assert np.isclose(porosity, expected)

    def test_porosity_plus_solid_equals_one(self, metrics, sample_voxels):
        """Test that porosity + solid fraction = 1 for binary."""
        porosity = metrics.porosity(sample_voxels, pore_id=0)
        solid_fraction = metrics.volume_fraction(sample_voxels, phase_id=1)

        assert np.isclose(porosity + solid_fraction, 1.0)

    def test_specific_surface_area(self, metrics, sample_voxels):
        """Test specific surface area calculation."""
        ssa = metrics.specific_surface_area(sample_voxels, phase_id=1)

        # Should be positive
        assert ssa > 0

        # For a cube, SSA = surface area / volume
        # Surface = 6 * 16^2 = 1536 (interior faces counted)
        # Volume = 16^3 = 4096
        # But we also have boundary effects

    def test_specific_surface_area_larger_object(self, metrics):
        """Test that SSA decreases with object size."""
        # Small cube
        small = np.zeros((32, 32, 32), dtype=np.uint8)
        small[12:20, 12:20, 12:20] = 1  # 8x8x8

        # Large cube
        large = np.zeros((32, 32, 32), dtype=np.uint8)
        large[4:28, 4:28, 4:28] = 1  # 24x24x24

        ssa_small = metrics.specific_surface_area(small, phase_id=1)
        ssa_large = metrics.specific_surface_area(large, phase_id=1)

        # Smaller objects have higher SSA
        assert ssa_small > ssa_large

    def test_tortuosity_straight_path(self, metrics):
        """Test tortuosity of straight path (should be ~1)."""
        # Create straight channel
        voxels = np.zeros((32, 32, 32), dtype=np.uint8)
        voxels[14:18, 14:18, :] = 1  # Straight channel in z

        tau = metrics.tortuosity_geometric(voxels, phase_id=1, direction="z")

        # Straight path should have tortuosity close to 1
        assert tau >= 1.0
        assert tau < 2.0  # Should be low

    def test_tortuosity_no_path(self, metrics):
        """Test tortuosity with no conducting path."""
        # Create isolated objects (no path through)
        voxels = np.zeros((32, 32, 32), dtype=np.uint8)
        voxels[5:10, 5:10, 5:10] = 1  # Isolated cube

        tau = metrics.tortuosity_geometric(voxels, phase_id=1, direction="z")

        # Should be infinite or very high
        assert tau > 10 or tau == float("inf")

    def test_connectivity_single_component(self, metrics, sample_voxels):
        """Test connectivity of single connected component."""
        conn = metrics.connectivity(sample_voxels, phase_id=1)

        assert conn["num_components"] == 1
        assert conn["connectivity_ratio"] == 1.0

    def test_connectivity_multiple_components(self, metrics, multiphase_voxels):
        """Test connectivity with multiple phases."""
        # Phase 1 and 2 are separate
        conn1 = metrics.connectivity(multiphase_voxels, phase_id=1)
        conn2 = metrics.connectivity(multiphase_voxels, phase_id=2)

        assert conn1["num_components"] == 1
        assert conn2["num_components"] == 1

    def test_particle_size_distribution(self, metrics, sample_voxels):
        """Test particle size distribution."""
        bin_edges, counts = metrics.particle_size_distribution(sample_voxels, phase_id=1)

        assert len(bin_edges) > 0
        assert len(counts) > 0
        assert len(counts) == len(bin_edges) - 1

    def test_interface_area(self, metrics, multiphase_voxels):
        """Test interface area calculation."""
        interface = metrics.interface_area(multiphase_voxels, phase1_id=1, phase2_id=0)

        # Should be positive (active material contacts pore)
        assert interface > 0

    def test_interface_area_no_contact(self, metrics, multiphase_voxels):
        """Test interface area when phases don't touch."""
        # Phase 1 and 2 are separated
        interface = metrics.interface_area(multiphase_voxels, phase1_id=1, phase2_id=2)

        # Should be zero or very small
        assert interface == 0

    def test_calculate_all(self, metrics, sample_voxels):
        """Test comprehensive metrics calculation."""
        all_metrics = metrics.calculate_all(sample_voxels, active_material_id=1, pore_id=0)

        # Check that all expected metrics are present
        assert "porosity" in all_metrics
        assert "active_material_fraction" in all_metrics
        assert "specific_surface_area" in all_metrics
        assert "tortuosity_x" in all_metrics
        assert "tortuosity_y" in all_metrics
        assert "tortuosity_z" in all_metrics
        assert "num_particles" in all_metrics
        assert "connectivity_ratio" in all_metrics


class TestMetricsComparator:
    """Tests for MetricsComparator class."""

    def test_compare(self):
        """Test metrics comparison."""
        reference = {
            "porosity": 0.4,
            "tortuosity": 1.5,
            "ssa": 1000.0,
        }
        generated = {
            "porosity": 0.42,
            "tortuosity": 1.6,
            "ssa": 980.0,
        }

        comparison = MetricsComparator.compare(generated, reference)

        assert "porosity" in comparison
        assert comparison["porosity"]["reference"] == 0.4
        assert comparison["porosity"]["generated"] == 0.42
        assert np.isclose(comparison["porosity"]["absolute_error"], 0.02)
        assert np.isclose(comparison["porosity"]["relative_error"], 0.05)

    def test_compare_missing_key(self):
        """Test comparison with missing keys."""
        reference = {"porosity": 0.4, "tortuosity": 1.5}
        generated = {"porosity": 0.42}  # Missing tortuosity

        comparison = MetricsComparator.compare(generated, reference)

        assert "porosity" in comparison
        assert "tortuosity" not in comparison  # Not compared

    def test_compare_zero_reference(self):
        """Test comparison when reference is zero."""
        reference = {"value": 0.0}
        generated = {"value": 0.1}

        comparison = MetricsComparator.compare(generated, reference)

        assert comparison["value"]["relative_error"] == float("inf")

    def test_summary(self):
        """Test summary generation."""
        reference = {"porosity": 0.4, "tortuosity": 1.5}
        generated = {"porosity": 0.42, "tortuosity": 1.55}

        comparison = MetricsComparator.compare(generated, reference)
        summary = MetricsComparator.summary(comparison)

        assert "Metrics Comparison Summary" in summary
        assert "porosity" in summary
        assert "tortuosity" in summary
        assert "Average Relative Error" in summary
