"""
Voxel to mesh conversion using Marching Cubes algorithm.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    from skimage import measure
except ImportError:
    measure = None

try:
    import trimesh
except ImportError:
    trimesh = None

from electrode_generator.config import PostprocessingConfig


class VoxelToMesh:
    """
    Convert voxel data to surface mesh using Marching Cubes.

    Args:
        config: Postprocessing configuration
    """

    def __init__(self, config: Optional[PostprocessingConfig] = None):
        self.config = config or PostprocessingConfig()

        if measure is None:
            raise ImportError("scikit-image is required. Install with: pip install scikit-image")

    def convert(
        self,
        voxels: np.ndarray,
        phase_id: int = 1,
        level: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert voxel volume to mesh.

        Args:
            voxels: 3D voxel array
            phase_id: Phase to extract (for segmented volumes)
            level: Isosurface level

        Returns:
            Tuple of (vertices, faces) arrays
        """
        level = level or self.config.level

        # Extract binary mask for the target phase
        if voxels.dtype in [np.uint8, np.int32, np.int64]:
            volume = (voxels == phase_id).astype(np.float32)
        else:
            volume = voxels.astype(np.float32)

        # Apply Marching Cubes
        spacing = (self.config.voxel_size,) * 3

        verts, faces, normals, values = measure.marching_cubes(
            volume,
            level=level,
            spacing=spacing,
            step_size=self.config.step_size,
            allow_degenerate=False,
        )

        return verts, faces

    def convert_multiphase(
        self,
        voxels: np.ndarray,
    ) -> dict:
        """
        Convert multi-phase voxel volume to separate meshes.

        Args:
            voxels: 3D voxel array with integer labels

        Returns:
            Dictionary mapping phase_id to (vertices, faces) tuples
        """
        unique_phases = np.unique(voxels)
        meshes = {}

        for phase_id in unique_phases:
            if phase_id == 0:  # Skip background
                continue

            verts, faces = self.convert(voxels, phase_id=int(phase_id))

            if len(verts) > 0:
                meshes[int(phase_id)] = (verts, faces)

        return meshes

    def to_trimesh(
        self,
        voxels: np.ndarray,
        phase_id: int = 1,
    ) -> "trimesh.Trimesh":
        """
        Convert voxels to trimesh object.

        Args:
            voxels: 3D voxel array
            phase_id: Phase to extract

        Returns:
            Trimesh mesh object
        """
        if trimesh is None:
            raise ImportError("trimesh is required. Install with: pip install trimesh")

        verts, faces = self.convert(voxels, phase_id=phase_id)
        return trimesh.Trimesh(vertices=verts, faces=faces)

    def export_stl(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        path: Union[str, Path],
    ) -> None:
        """
        Export mesh to STL file.

        Args:
            vertices: Vertex coordinates
            faces: Face indices
            path: Output file path
        """
        if trimesh is None:
            raise ImportError("trimesh is required for STL export")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(str(path), file_type="stl")

    def export_obj(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        path: Union[str, Path],
    ) -> None:
        """
        Export mesh to OBJ file.

        Args:
            vertices: Vertex coordinates
            faces: Face indices
            path: Output file path
        """
        if trimesh is None:
            raise ImportError("trimesh is required for OBJ export")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(str(path), file_type="obj")


class MeshRefinement:
    """
    Mesh refinement and repair operations.
    """

    def __init__(self, config: Optional[PostprocessingConfig] = None):
        self.config = config or PostprocessingConfig()

        if trimesh is None:
            raise ImportError("trimesh is required. Install with: pip install trimesh")

    def smooth(
        self,
        mesh: "trimesh.Trimesh",
        iterations: Optional[int] = None,
        lamb: Optional[float] = None,
    ) -> "trimesh.Trimesh":
        """
        Apply Laplacian smoothing.

        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            lamb: Smoothing factor

        Returns:
            Smoothed mesh
        """
        iterations = iterations or self.config.smooth_iterations
        lamb = lamb or self.config.smooth_factor

        trimesh.smoothing.filter_laplacian(
            mesh,
            iterations=iterations,
            lamb=lamb,
        )

        return mesh

    def simplify(
        self,
        mesh: "trimesh.Trimesh",
        target_faces: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> "trimesh.Trimesh":
        """
        Simplify mesh by reducing face count.

        Args:
            mesh: Input mesh
            target_faces: Target number of faces
            ratio: Simplification ratio (0-1), e.g., 0.5 means keep 50% of faces

        Returns:
            Simplified mesh
        """
        current_faces = len(mesh.faces)

        if target_faces:
            ratio = target_faces / current_faces
        elif ratio is None:
            ratio = 0.5

        # Ensure ratio is in valid range
        ratio = max(0.01, min(0.99, ratio))

        # Try quadric decimation (requires fast_simplification)
        try:
            # trimesh 4.x API: percent is the fraction of faces to keep
            return mesh.simplify_quadric_decimation(percent=ratio)
        except (ImportError, ModuleNotFoundError):
            # Fallback: return a copy
            return mesh.copy()
        except TypeError:
            # Old API: target face count
            try:
                return mesh.simplify_quadric_decimation(int(current_faces * ratio))
            except Exception:
                return mesh.copy()

    def repair(self, mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
        """
        Repair mesh (fix normals, remove duplicates, fill holes).

        Args:
            mesh: Input mesh

        Returns:
            Repaired mesh
        """
        # Remove duplicate vertices
        mesh.merge_vertices()

        # Remove degenerate faces using nondegenerate_faces mask
        if hasattr(mesh, 'nondegenerate_faces'):
            mask = mesh.nondegenerate_faces()
            if not mask.all():
                mesh.update_faces(mask)
        elif hasattr(mesh, 'remove_degenerate_faces'):
            # Fallback for older trimesh versions
            mesh.remove_degenerate_faces()

        # Fix normals
        mesh.fix_normals()

        # Fill holes
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)

        return mesh

    def validate(self, mesh: "trimesh.Trimesh") -> dict:
        """
        Validate mesh for simulation compatibility.

        Args:
            mesh: Input mesh

        Returns:
            Validation results dictionary
        """
        return {
            "is_watertight": mesh.is_watertight,
            "is_volume": mesh.is_volume,
            "is_winding_consistent": mesh.is_winding_consistent,
            "euler_number": mesh.euler_number,
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "volume": mesh.volume if mesh.is_watertight else None,
            "surface_area": mesh.area,
            "bounding_box": mesh.bounds.tolist(),
        }

    def optimize_for_comsol(
        self,
        mesh: "trimesh.Trimesh",
    ) -> "trimesh.Trimesh":
        """
        Optimize mesh for COMSOL import.

        Args:
            mesh: Input mesh

        Returns:
            Optimized mesh
        """
        # Repair first
        mesh = self.repair(mesh)

        # Smooth for better element quality
        if self.config.smooth_iterations > 0:
            mesh = self.smooth(mesh)

        # Ensure watertight
        if not mesh.is_watertight:
            try:
                import pymeshfix
                meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
                meshfix.repair(verbose=False)
                mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
            except ImportError:
                pass  # pymeshfix not available

        return mesh


class TPMSGenerator:
    """
    Generate TPMS (Triply Periodic Minimal Surfaces) structures.

    Supports: Gyroid, Schwarz P, Schwarz D, I-WP, Neovius
    """

    TPMS_FUNCTIONS = {
        "gyroid": lambda x, y, z: np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x),
        "schwarz_p": lambda x, y, z: np.cos(x) + np.cos(y) + np.cos(z),
        "schwarz_d": lambda x, y, z: (
            np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z)
        ),
        "iwp": lambda x, y, z: (
            np.cos(x) * np.cos(y)
            + np.cos(y) * np.cos(z)
            + np.cos(z) * np.cos(x)
            - np.cos(x) * np.cos(y) * np.cos(z)
        ),
        "neovius": lambda x, y, z: (
            3 * (np.cos(x) + np.cos(y) + np.cos(z))
            + 4 * np.cos(x) * np.cos(y) * np.cos(z)
        ),
    }

    def generate(
        self,
        tpms_type: str = "gyroid",
        resolution: int = 100,
        periods: int = 2,
        thickness: float = 0.3,
        dimensions: Tuple[float, float, float] = (10.0, 10.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate TPMS structure.

        Args:
            tpms_type: Type of TPMS structure
            resolution: Voxel resolution
            periods: Number of periods
            thickness: Wall thickness (0-1)
            dimensions: Physical dimensions (mm)

        Returns:
            Tuple of (vertices, faces)
        """
        if measure is None:
            raise ImportError("scikit-image is required")

        if tpms_type not in self.TPMS_FUNCTIONS:
            raise ValueError(f"Unknown TPMS type: {tpms_type}. Available: {list(self.TPMS_FUNCTIONS.keys())}")

        func = self.TPMS_FUNCTIONS[tpms_type]

        # Calculate resolution for each dimension
        max_dim = max(dimensions)
        res_x = int(resolution * dimensions[0] / max_dim)
        res_y = int(resolution * dimensions[1] / max_dim)
        res_z = int(resolution * dimensions[2] / max_dim)

        # Create coordinate grids
        x = np.linspace(0, 2 * np.pi * periods, res_x)
        y = np.linspace(0, 2 * np.pi * periods, res_y)
        z = np.linspace(0, 2 * np.pi * periods, res_z)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Evaluate TPMS function
        field = func(X, Y, Z)

        # Create sheet structure
        volume = np.abs(field) < thickness

        # Marching cubes
        spacing = (
            dimensions[0] / res_x,
            dimensions[1] / res_y,
            dimensions[2] / res_z,
        )

        verts, faces, normals, values = measure.marching_cubes(
            volume.astype(float),
            level=0.5,
            spacing=spacing,
        )

        return verts, faces

    def generate_mesh(
        self,
        tpms_type: str = "gyroid",
        **kwargs,
    ) -> "trimesh.Trimesh":
        """
        Generate TPMS structure as trimesh object.

        Args:
            tpms_type: Type of TPMS structure
            **kwargs: Additional arguments for generate()

        Returns:
            Trimesh mesh object
        """
        if trimesh is None:
            raise ImportError("trimesh is required")

        verts, faces = self.generate(tpms_type, **kwargs)
        return trimesh.Trimesh(vertices=verts, faces=faces)
