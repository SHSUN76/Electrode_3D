"""
Microstructure metrics calculation for electrode analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from scipy import ndimage
    from scipy.ndimage import label, distance_transform_edt
except ImportError:
    ndimage = None


class MicrostructureMetrics:
    """
    Calculate microstructure metrics for electrode analysis.

    Supports:
    - Volume fraction
    - Porosity
    - Specific surface area
    - Tortuosity (geometric approximation)
    - Connectivity analysis
    - Particle size distribution
    """

    def __init__(self, voxel_size: float = 1.0):
        """
        Initialize metrics calculator.

        Args:
            voxel_size: Physical size of each voxel (micrometers)
        """
        self.voxel_size = voxel_size

        if ndimage is None:
            raise ImportError("scipy is required. Install with: pip install scipy")

    def volume_fraction(
        self,
        voxels: np.ndarray,
        phase_id: int = 1,
    ) -> float:
        """
        Calculate volume fraction of a phase.

        Args:
            voxels: 3D voxel array with phase labels
            phase_id: Phase ID to calculate fraction for

        Returns:
            Volume fraction (0-1)
        """
        return np.sum(voxels == phase_id) / voxels.size

    def porosity(
        self,
        voxels: np.ndarray,
        pore_id: int = 0,
    ) -> float:
        """
        Calculate porosity (void fraction).

        Args:
            voxels: 3D voxel array
            pore_id: Phase ID representing pores

        Returns:
            Porosity (0-1)
        """
        return self.volume_fraction(voxels, pore_id)

    def specific_surface_area(
        self,
        voxels: np.ndarray,
        phase_id: int = 1,
    ) -> float:
        """
        Calculate specific surface area using voxel face counting.

        Args:
            voxels: 3D voxel array
            phase_id: Phase ID to analyze

        Returns:
            Specific surface area (1/length unit)
        """
        binary = (voxels == phase_id).astype(np.uint8)

        # Count exposed faces in each direction
        surface_voxels = 0

        # X direction
        diff_x = np.abs(np.diff(binary, axis=0))
        surface_voxels += np.sum(diff_x)

        # Y direction
        diff_y = np.abs(np.diff(binary, axis=1))
        surface_voxels += np.sum(diff_y)

        # Z direction
        diff_z = np.abs(np.diff(binary, axis=2))
        surface_voxels += np.sum(diff_z)

        # Add boundary faces
        surface_voxels += np.sum(binary[0, :, :])
        surface_voxels += np.sum(binary[-1, :, :])
        surface_voxels += np.sum(binary[:, 0, :])
        surface_voxels += np.sum(binary[:, -1, :])
        surface_voxels += np.sum(binary[:, :, 0])
        surface_voxels += np.sum(binary[:, :, -1])

        # Surface area = faces * face area
        face_area = self.voxel_size ** 2
        total_surface = surface_voxels * face_area

        # Volume of phase
        phase_volume = np.sum(binary) * (self.voxel_size ** 3)

        if phase_volume == 0:
            return 0.0

        return total_surface / phase_volume

    def tortuosity_geometric(
        self,
        voxels: np.ndarray,
        phase_id: int = 1,
        direction: str = "z",
    ) -> float:
        """
        Calculate geometric tortuosity using shortest path approximation.

        Args:
            voxels: 3D voxel array
            phase_id: Conducting phase ID
            direction: Direction for tortuosity ("x", "y", or "z")

        Returns:
            Tortuosity factor (>= 1)
        """
        binary = (voxels == phase_id).astype(np.uint8)

        # Get dimension indices
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis = axis_map.get(direction, 2)

        # Create distance transform from starting face
        shape = binary.shape

        # Starting mask (first slice in direction)
        start_mask = np.zeros_like(binary)
        if axis == 0:
            start_mask[0, :, :] = binary[0, :, :]
        elif axis == 1:
            start_mask[:, 0, :] = binary[:, 0, :]
        else:
            start_mask[:, :, 0] = binary[:, :, 0]

        if np.sum(start_mask) == 0:
            return float("inf")  # No conducting path

        # Distance transform through conducting phase
        # Invert: 0 where conducting, inf where not
        cost = np.where(binary, 0, np.inf)

        # Use geodesic distance transform approximation
        dist = distance_transform_edt(binary)

        # Get ending face distances
        if axis == 0:
            end_distances = dist[-1, :, :]
            straight_distance = shape[0]
        elif axis == 1:
            end_distances = dist[:, -1, :]
            straight_distance = shape[1]
        else:
            end_distances = dist[:, :, -1]
            straight_distance = shape[2]

        # Average path length
        valid_distances = end_distances[end_distances > 0]

        if len(valid_distances) == 0:
            return float("inf")

        avg_path_length = np.mean(valid_distances)
        straight_distance *= self.voxel_size

        # Tortuosity = actual path / straight path
        # Approximation using distance transform
        tortuosity = (avg_path_length * self.voxel_size) / straight_distance

        # Tortuosity should be >= 1
        return max(1.0, tortuosity)

    def connectivity(
        self,
        voxels: np.ndarray,
        phase_id: int = 1,
    ) -> Dict[str, Union[int, float, List[int]]]:
        """
        Analyze connectivity of a phase.

        Args:
            voxels: 3D voxel array
            phase_id: Phase ID to analyze

        Returns:
            Dictionary with connectivity metrics
        """
        binary = (voxels == phase_id).astype(np.uint8)

        # Label connected components
        labeled, num_features = label(binary)

        # Component sizes
        component_sizes = []
        for i in range(1, num_features + 1):
            size = np.sum(labeled == i)
            component_sizes.append(size)

        component_sizes = sorted(component_sizes, reverse=True)

        # Calculate connectivity ratio
        total_voxels = np.sum(binary)
        if total_voxels == 0:
            connectivity_ratio = 0.0
        elif len(component_sizes) > 0:
            connectivity_ratio = component_sizes[0] / total_voxels
        else:
            connectivity_ratio = 0.0

        return {
            "num_components": num_features,
            "largest_component_size": component_sizes[0] if component_sizes else 0,
            "connectivity_ratio": connectivity_ratio,
            "component_sizes": component_sizes[:10],  # Top 10
        }

    def particle_size_distribution(
        self,
        voxels: np.ndarray,
        phase_id: int = 1,
        num_bins: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate particle size distribution using distance transform.

        Args:
            voxels: 3D voxel array
            phase_id: Phase ID to analyze
            num_bins: Number of histogram bins

        Returns:
            Tuple of (bin_edges, histogram_counts)
        """
        binary = (voxels == phase_id).astype(np.uint8)

        # Distance transform gives local thickness
        dist = distance_transform_edt(binary)

        # Get non-zero distances (inside particles)
        local_thickness = dist[binary > 0]

        if len(local_thickness) == 0:
            return np.array([]), np.array([])

        # Convert to physical units and double (diameter)
        diameters = local_thickness * 2 * self.voxel_size

        # Histogram
        counts, bin_edges = np.histogram(diameters, bins=num_bins)

        return bin_edges, counts

    def interface_area(
        self,
        voxels: np.ndarray,
        phase1_id: int = 1,
        phase2_id: int = 2,
    ) -> float:
        """
        Calculate interface area between two phases.

        Args:
            voxels: 3D voxel array
            phase1_id: First phase ID
            phase2_id: Second phase ID

        Returns:
            Interface area (in voxel_size^2 units)
        """
        binary1 = (voxels == phase1_id).astype(np.uint8)
        binary2 = (voxels == phase2_id).astype(np.uint8)

        # Count adjacent voxel pairs
        interface_count = 0

        # X direction
        interface_count += np.sum(binary1[:-1, :, :] & binary2[1:, :, :])
        interface_count += np.sum(binary2[:-1, :, :] & binary1[1:, :, :])

        # Y direction
        interface_count += np.sum(binary1[:, :-1, :] & binary2[:, 1:, :])
        interface_count += np.sum(binary2[:, :-1, :] & binary1[:, 1:, :])

        # Z direction
        interface_count += np.sum(binary1[:, :, :-1] & binary2[:, :, 1:])
        interface_count += np.sum(binary2[:, :, :-1] & binary1[:, :, 1:])

        return interface_count * (self.voxel_size ** 2)

    def calculate_all(
        self,
        voxels: np.ndarray,
        active_material_id: int = 1,
        pore_id: int = 0,
        binder_id: Optional[int] = 2,
    ) -> Dict[str, float]:
        """
        Calculate all metrics for electrode microstructure.

        Args:
            voxels: 3D voxel array with phase labels
            active_material_id: Active material phase ID
            pore_id: Pore phase ID
            binder_id: Binder phase ID (optional)

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # Volume fractions
        metrics["porosity"] = self.porosity(voxels, pore_id)
        metrics["active_material_fraction"] = self.volume_fraction(voxels, active_material_id)

        if binder_id is not None:
            metrics["binder_fraction"] = self.volume_fraction(voxels, binder_id)

        # Surface areas
        metrics["specific_surface_area"] = self.specific_surface_area(voxels, active_material_id)

        # Tortuosity in all directions
        metrics["tortuosity_x"] = self.tortuosity_geometric(voxels, active_material_id, "x")
        metrics["tortuosity_y"] = self.tortuosity_geometric(voxels, active_material_id, "y")
        metrics["tortuosity_z"] = self.tortuosity_geometric(voxels, active_material_id, "z")
        metrics["tortuosity_avg"] = np.mean([
            metrics["tortuosity_x"],
            metrics["tortuosity_y"],
            metrics["tortuosity_z"],
        ])

        # Connectivity
        connectivity = self.connectivity(voxels, active_material_id)
        metrics["num_particles"] = connectivity["num_components"]
        metrics["connectivity_ratio"] = connectivity["connectivity_ratio"]

        # Interface areas
        metrics["am_pore_interface"] = self.interface_area(voxels, active_material_id, pore_id)

        if binder_id is not None:
            metrics["am_binder_interface"] = self.interface_area(voxels, active_material_id, binder_id)

        return metrics


class MetricsComparator:
    """
    Compare metrics between generated and reference structures.
    """

    @staticmethod
    def compare(
        generated_metrics: Dict[str, float],
        reference_metrics: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare generated metrics against reference.

        Args:
            generated_metrics: Metrics from generated structure
            reference_metrics: Metrics from reference structure

        Returns:
            Comparison results with absolute and relative errors
        """
        comparison = {}

        for key in reference_metrics:
            if key in generated_metrics:
                ref_val = reference_metrics[key]
                gen_val = generated_metrics[key]

                abs_error = abs(gen_val - ref_val)

                if ref_val != 0:
                    rel_error = abs_error / abs(ref_val)
                else:
                    rel_error = float("inf") if gen_val != 0 else 0.0

                comparison[key] = {
                    "reference": ref_val,
                    "generated": gen_val,
                    "absolute_error": abs_error,
                    "relative_error": rel_error,
                }

        return comparison

    @staticmethod
    def summary(comparison: Dict[str, Dict[str, float]]) -> str:
        """
        Generate summary text from comparison.

        Args:
            comparison: Comparison results from compare()

        Returns:
            Formatted summary string
        """
        lines = ["Metrics Comparison Summary", "=" * 40]

        for key, values in comparison.items():
            lines.append(f"\n{key}:")
            lines.append(f"  Reference: {values['reference']:.4f}")
            lines.append(f"  Generated: {values['generated']:.4f}")
            lines.append(f"  Rel. Error: {values['relative_error']*100:.2f}%")

        # Overall score
        avg_rel_error = np.mean([v["relative_error"] for v in comparison.values()])
        lines.append(f"\n{'=' * 40}")
        lines.append(f"Average Relative Error: {avg_rel_error*100:.2f}%")

        return "\n".join(lines)
