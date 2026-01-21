"""
Mesh export utilities for various simulation tools.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

from electrode_generator.config import PostprocessingConfig


class MeshExporter:
    """
    Export meshes to various formats for simulation tools.

    Supports:
    - STL (COMSOL, OpenFOAM, FEniCS)
    - OBJ (Blender, general)
    - VTK (ParaView, PyVista)
    - NASTRAN (COMSOL import)
    """

    def __init__(self, config: Optional[PostprocessingConfig] = None):
        self.config = config or PostprocessingConfig()

    def export(
        self,
        mesh: "trimesh.Trimesh",
        path: Union[str, Path],
        format: Optional[str] = None,
    ) -> Path:
        """
        Export mesh to file.

        Args:
            mesh: Trimesh mesh object
            path: Output file path
            format: File format (inferred from extension if None)

        Returns:
            Path to exported file
        """
        if trimesh is None:
            raise ImportError("trimesh is required")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = path.suffix.lower().lstrip(".")

        mesh.export(str(path), file_type=format)
        return path

    def export_stl_binary(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        path: Union[str, Path],
    ) -> Path:
        """
        Export to binary STL (smaller file size).

        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3)
            path: Output file path

        Returns:
            Path to exported file
        """
        if trimesh is None:
            raise ImportError("trimesh is required")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return self.export(mesh, path, format="stl")

    def export_stl_ascii(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        path: Union[str, Path],
        name: str = "electrode",
    ) -> Path:
        """
        Export to ASCII STL (human-readable, larger file).

        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3)
            path: Output file path
            name: Solid name in STL

        Returns:
            Path to exported file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms > 0, norms, 1)

        with open(path, "w") as f:
            f.write(f"solid {name}\n")

            for i, face in enumerate(faces):
                n = normals[i]
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")

                for vi in face:
                    v = vertices[vi]
                    f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")

                f.write("    endloop\n")
                f.write("  endfacet\n")

            f.write(f"endsolid {name}\n")

        return path

    def export_obj(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        path: Union[str, Path],
        vertex_normals: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Export to OBJ format.

        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3)
            path: Output file path
            vertex_normals: Optional vertex normals

        Returns:
            Path to exported file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write("# Electrode 3D Generator OBJ Export\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n\n")

            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write normals if provided
            if vertex_normals is not None:
                f.write("\n")
                for n in vertex_normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # Write faces (OBJ uses 1-based indexing)
            f.write("\n")
            for face in faces:
                if vertex_normals is not None:
                    f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        return path

    def export_vtk(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        path: Union[str, Path],
        scalars: Optional[Dict[str, np.ndarray]] = None,
    ) -> Path:
        """
        Export to VTK format for ParaView.

        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3)
            path: Output file path
            scalars: Optional scalar data per vertex

        Returns:
            Path to exported file
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("pyvista is required for VTK export")

        # Create faces array with count prefix
        cells = np.hstack([
            np.full((len(faces), 1), 3),
            faces
        ]).flatten()

        mesh = pv.PolyData(vertices, cells)

        # Add scalar data
        if scalars:
            for name, data in scalars.items():
                mesh[name] = data

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mesh.save(str(path))
        return path

    def export_for_comsol(
        self,
        mesh: "trimesh.Trimesh",
        path: Union[str, Path],
        scale: float = 1e-3,  # mm to m
        format: str = "stl",
    ) -> Path:
        """
        Export mesh optimized for COMSOL import.

        Applies:
        - Scale conversion (mm to m)
        - Coordinate system adjustment
        - Mesh validation

        Args:
            mesh: Trimesh mesh object
            path: Output file path
            scale: Scale factor (default: mm to m)
            format: Export format ("stl" or "nastran")

        Returns:
            Path to exported file
        """
        if trimesh is None:
            raise ImportError("trimesh is required")

        # Apply scale
        scaled_mesh = mesh.copy()
        scaled_mesh.apply_scale(scale)

        # Ensure watertight
        if not scaled_mesh.is_watertight:
            try:
                import pymeshfix
                meshfix = pymeshfix.MeshFix(scaled_mesh.vertices, scaled_mesh.faces)
                meshfix.repair(verbose=False)
                scaled_mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
            except ImportError:
                pass

        # Export in requested format
        if format.lower() == "nastran":
            return self.export_nastran(
                scaled_mesh.vertices,
                scaled_mesh.faces,
                path
            )
        else:
            return self.export(scaled_mesh, path, format="stl")

    def export_nastran(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        path: Union[str, Path],
        title: str = "Electrode Microstructure",
    ) -> Path:
        """
        Export mesh in NASTRAN format for COMSOL import.

        NASTRAN format is well-supported by COMSOL and preserves
        mesh topology better than STL for some import cases.

        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3)
            path: Output file path
            title: Title for NASTRAN file

        Returns:
            Path to exported file
        """
        path = Path(path)
        if not path.suffix.lower() in ['.nas', '.nastran', '.bdf']:
            path = path.with_suffix('.nas')

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # Header
            f.write(f"$ NASTRAN mesh exported by Electrode 3D Generator\n")
            f.write(f"$ Title: {title}\n")
            f.write(f"$ Vertices: {len(vertices)}, Faces: {len(faces)}\n")
            f.write("BEGIN BULK\n")

            # Write GRID cards (vertices)
            # Format: GRID, ID, CP, X, Y, Z, CD, PS, SEID
            for i, v in enumerate(vertices, start=1):
                # Use large field format for precision
                f.write(f"GRID*   {i:16d}                {v[0]:16.8E}{v[1]:16.8E}\n")
                f.write(f"*       {v[2]:16.8E}\n")

            # Write CTRIA3 cards (triangular elements)
            # Format: CTRIA3, EID, PID, G1, G2, G3
            pid = 1  # Property ID
            for i, face in enumerate(faces, start=1):
                # NASTRAN uses 1-based indexing
                g1, g2, g3 = face[0] + 1, face[1] + 1, face[2] + 1
                f.write(f"CTRIA3  {i:8d}{pid:8d}{g1:8d}{g2:8d}{g3:8d}\n")

            # Shell property (for surface mesh)
            f.write(f"PSHELL  {pid:8d}       1      1.0       1              1\n")

            # Material (placeholder)
            f.write("MAT1           1   2.1+11             0.3     7800.\n")

            f.write("ENDDATA\n")

        return path

    def export_gmsh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        path: Union[str, Path],
    ) -> Path:
        """
        Export mesh in Gmsh MSH format.

        Gmsh format is useful for further mesh refinement with Gmsh
        before importing to COMSOL.

        Args:
            vertices: Vertex coordinates (N, 3)
            faces: Face indices (M, 3)
            path: Output file path

        Returns:
            Path to exported file
        """
        path = Path(path)
        if not path.suffix.lower() in ['.msh']:
            path = path.with_suffix('.msh')

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # Header
            f.write("$MeshFormat\n")
            f.write("2.2 0 8\n")
            f.write("$EndMeshFormat\n")

            # Nodes
            f.write("$Nodes\n")
            f.write(f"{len(vertices)}\n")
            for i, v in enumerate(vertices, start=1):
                f.write(f"{i} {v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n")
            f.write("$EndNodes\n")

            # Elements (triangles)
            f.write("$Elements\n")
            f.write(f"{len(faces)}\n")
            for i, face in enumerate(faces, start=1):
                # Element type 2 = 3-node triangle
                # Format: elem-number elem-type num-tags tags... node-list
                g1, g2, g3 = face[0] + 1, face[1] + 1, face[2] + 1
                f.write(f"{i} 2 0 {g1} {g2} {g3}\n")
            f.write("$EndElements\n")

        return path

    def export_batch(
        self,
        meshes: Dict[str, "trimesh.Trimesh"],
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
    ) -> Dict[str, List[Path]]:
        """
        Batch export multiple meshes.

        Args:
            meshes: Dictionary mapping names to meshes
            output_dir: Output directory
            formats: List of formats to export

        Returns:
            Dictionary mapping names to list of exported paths
        """
        formats = formats or self.config.export_formats
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for name, mesh in meshes.items():
            results[name] = []
            for fmt in formats:
                path = output_dir / f"{name}.{fmt}"
                self.export(mesh, path, format=fmt)
                results[name].append(path)

        return results
