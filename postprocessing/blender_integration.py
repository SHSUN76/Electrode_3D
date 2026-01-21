"""
Blender MCP integration for mesh refinement.

This module provides a Python API wrapper for Blender MCP operations,
enabling automated mesh refinement in the electrode generation pipeline.

Requires:
- Blender with MCP addon running on port 9876
- blender-mcp server connected to Claude Code
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


logger = logging.getLogger(__name__)


@dataclass
class MeshQualityReport:
    """Mesh quality analysis report."""

    is_manifold: bool
    is_watertight: bool
    non_manifold_edges: int
    non_manifold_verts: int
    degenerate_faces: int
    isolated_verts: int
    volume: Optional[float]
    surface_area: float
    bounding_box: List[List[float]]
    vertex_count: int
    face_count: int

    @property
    def is_simulation_ready(self) -> bool:
        """Check if mesh is ready for FEM simulation."""
        return (
            self.is_manifold
            and self.is_watertight
            and self.non_manifold_edges == 0
            and self.degenerate_faces == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_manifold": self.is_manifold,
            "is_watertight": self.is_watertight,
            "non_manifold_edges": self.non_manifold_edges,
            "non_manifold_verts": self.non_manifold_verts,
            "degenerate_faces": self.degenerate_faces,
            "isolated_verts": self.isolated_verts,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "bounding_box": self.bounding_box,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "is_simulation_ready": self.is_simulation_ready,
        }


class BlenderMeshRefiner:
    """
    Blender MCP wrapper for automated mesh refinement.

    This class provides high-level methods to:
    - Import STL/OBJ meshes into Blender
    - Analyze mesh quality (manifold, watertight checks)
    - Apply Voxel Remesh for water-tight mesh generation
    - Export refined meshes for COMSOL simulation

    Note:
        This class generates Blender Python code that should be executed
        via the Blender MCP. When used in the Claude Code environment,
        the generated code is passed to mcp__blender__execute_blender_code.

    Example:
        >>> refiner = BlenderMeshRefiner()
        >>> code = refiner.get_import_code("electrode.stl")
        >>> # Execute via Blender MCP: mcp__blender__execute_blender_code(code=code)
    """

    def __init__(self, voxel_size: float = 2.0, smooth_iterations: int = 2):
        """
        Initialize the Blender mesh refiner.

        Args:
            voxel_size: Default voxel size for remeshing (in mm)
            smooth_iterations: Number of smoothing iterations after remesh
        """
        self.voxel_size = voxel_size
        self.smooth_iterations = smooth_iterations

    def get_import_code(self, stl_path: Union[str, Path], name: str = "electrode") -> str:
        """
        Generate Blender Python code to import an STL file.

        Args:
            stl_path: Path to STL file
            name: Object name in Blender

        Returns:
            Blender Python code string
        """
        stl_path = Path(stl_path).absolute()

        code = f'''
import bpy

# Clear default objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import STL
bpy.ops.wm.stl_import(filepath=r"{stl_path}")

# Rename imported object
if bpy.context.selected_objects:
    obj = bpy.context.selected_objects[0]
    obj.name = "{name}"
    bpy.context.view_layer.objects.active = obj

print(f"Imported: {{obj.name}} with {{len(obj.data.vertices)}} vertices, {{len(obj.data.polygons)}} faces")
'''
        return code.strip()

    def get_analyze_code(self, object_name: str = "electrode") -> str:
        """
        Generate Blender Python code to analyze mesh quality.

        Args:
            object_name: Name of the object to analyze

        Returns:
            Blender Python code string
        """
        code = f'''
import bpy
import bmesh
import json

obj = bpy.data.objects.get("{object_name}")
if obj is None:
    raise ValueError("Object '{object_name}' not found")

# Enter edit mode for analysis
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)

# Analyze mesh
non_manifold_edges = sum(1 for e in bm.edges if not e.is_manifold)
non_manifold_verts = sum(1 for v in bm.verts if not v.is_manifold)
degenerate_faces = sum(1 for f in bm.faces if f.calc_area() < 1e-8)
isolated_verts = sum(1 for v in bm.verts if not v.link_edges)

# Exit edit mode
bpy.ops.object.mode_set(mode='OBJECT')

# Get mesh statistics
mesh = obj.data
is_manifold = non_manifold_edges == 0 and non_manifold_verts == 0

# Calculate volume and surface area
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.object.mode_set(mode='OBJECT')

# Use Blender's built-in mesh analysis
depsgraph = bpy.context.evaluated_depsgraph_get()
obj_eval = obj.evaluated_get(depsgraph)
mesh_eval = obj_eval.to_mesh()

# Calculate bounds
bounds = [list(obj.bound_box[0]), list(obj.bound_box[6])]

# Attempt to calculate volume (only works for closed meshes)
bm_check = bmesh.new()
bm_check.from_mesh(mesh_eval)
bm_check.transform(obj.matrix_world)

try:
    volume = bm_check.calc_volume()
    is_watertight = True
except:
    volume = None
    is_watertight = False

surface_area = sum(f.calc_area() for f in bm_check.faces)
bm_check.free()
obj_eval.to_mesh_clear()

result = {{
    "is_manifold": is_manifold,
    "is_watertight": is_watertight,
    "non_manifold_edges": non_manifold_edges,
    "non_manifold_verts": non_manifold_verts,
    "degenerate_faces": degenerate_faces,
    "isolated_verts": isolated_verts,
    "volume": volume,
    "surface_area": surface_area,
    "bounding_box": bounds,
    "vertex_count": len(mesh.vertices),
    "face_count": len(mesh.polygons)
}}

print("MESH_QUALITY_RESULT:" + json.dumps(result))
'''
        return code.strip()

    def get_remesh_code(
        self,
        object_name: str = "electrode",
        voxel_size: Optional[float] = None,
        smooth: bool = True,
    ) -> str:
        """
        Generate Blender Python code for Voxel Remesh.

        Voxel Remesh creates a watertight, manifold mesh from any input,
        which is essential for FEM simulation in COMSOL.

        Args:
            object_name: Name of the object to remesh
            voxel_size: Voxel size in object units (smaller = more detail)
            smooth: Whether to apply smoothing after remesh

        Returns:
            Blender Python code string
        """
        voxel_size = voxel_size or self.voxel_size

        code = f'''
import bpy

obj = bpy.data.objects.get("{object_name}")
if obj is None:
    raise ValueError("Object '{object_name}' not found")

bpy.context.view_layer.objects.active = obj

# Add Remesh modifier
remesh = obj.modifiers.new(name="Remesh", type='REMESH')
remesh.mode = 'VOXEL'
remesh.voxel_size = {voxel_size}
remesh.use_smooth_shade = True

# Apply modifier
bpy.ops.object.modifier_apply(modifier="Remesh")

print(f"Applied Voxel Remesh with size {voxel_size}")
print(f"New mesh: {{len(obj.data.vertices)}} vertices, {{len(obj.data.polygons)}} faces")
'''

        if smooth:
            code += f'''
# Apply Laplacian Smooth
smooth = obj.modifiers.new(name="Smooth", type='LAPLACIANSMOOTH')
smooth.iterations = {self.smooth_iterations}
smooth.lambda_factor = 0.5
bpy.ops.object.modifier_apply(modifier="Smooth")
print(f"Applied Laplacian smoothing with {self.smooth_iterations} iterations")
'''
        return code.strip()

    def get_export_code(
        self,
        output_path: Union[str, Path],
        object_name: str = "electrode",
        format: str = "stl",
        scale: float = 1.0,
    ) -> str:
        """
        Generate Blender Python code to export refined mesh.

        Args:
            output_path: Output file path
            object_name: Name of the object to export
            format: Export format ("stl", "obj", "ply", "fbx")
            scale: Scale factor (e.g., 0.001 for mm to m conversion)

        Returns:
            Blender Python code string
        """
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        code = f'''
import bpy

obj = bpy.data.objects.get("{object_name}")
if obj is None:
    raise ValueError("Object '{object_name}' not found")

# Select only this object
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
'''

        if scale != 1.0:
            code += f'''
# Apply scale
obj.scale = ({scale}, {scale}, {scale})
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
'''

        if format.lower() == "stl":
            code += f'''
# Export as STL
bpy.ops.wm.stl_export(
    filepath=r"{output_path}",
    export_selected_objects=True,
    ascii_format=False
)
'''
        elif format.lower() == "obj":
            code += f'''
# Export as OBJ
bpy.ops.wm.obj_export(
    filepath=r"{output_path}",
    export_selected_objects=True,
    export_materials=False
)
'''
        elif format.lower() == "ply":
            code += f'''
# Export as PLY
bpy.ops.wm.ply_export(
    filepath=r"{output_path}",
    export_selected_objects=True
)
'''

        code += f'''
print(f"Exported to: {output_path}")
'''
        return code.strip()

    def get_full_refinement_code(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        voxel_size: Optional[float] = None,
        smooth: bool = True,
        scale_for_comsol: bool = True,
    ) -> str:
        """
        Generate complete Blender Python code for full mesh refinement pipeline.

        This combines import, remesh, and export into a single script.

        Args:
            input_path: Input STL/OBJ path
            output_path: Output file path
            voxel_size: Remesh voxel size
            smooth: Apply smoothing
            scale_for_comsol: Scale from mm to m for COMSOL

        Returns:
            Complete Blender Python code string
        """
        input_path = Path(input_path).absolute()
        output_path = Path(output_path).absolute()
        voxel_size = voxel_size or self.voxel_size
        scale = 0.001 if scale_for_comsol else 1.0

        code = f'''
import bpy
import bmesh

# ===== FULL MESH REFINEMENT PIPELINE =====

# 1. Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 2. Import mesh
bpy.ops.wm.stl_import(filepath=r"{input_path}")
obj = bpy.context.selected_objects[0]
obj.name = "electrode"
bpy.context.view_layer.objects.active = obj
print(f"Imported: {{len(obj.data.vertices)}} vertices, {{len(obj.data.polygons)}} faces")

# 3. Pre-refinement analysis
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
pre_non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
bpy.ops.object.mode_set(mode='OBJECT')
print(f"Pre-refinement non-manifold edges: {{pre_non_manifold}}")

# 4. Apply Voxel Remesh
remesh = obj.modifiers.new(name="Remesh", type='REMESH')
remesh.mode = 'VOXEL'
remesh.voxel_size = {voxel_size}
remesh.use_smooth_shade = True
bpy.ops.object.modifier_apply(modifier="Remesh")
print(f"Applied Voxel Remesh with size {voxel_size}")
'''

        if smooth:
            code += f'''
# 5. Apply smoothing
smooth = obj.modifiers.new(name="Smooth", type='LAPLACIANSMOOTH')
smooth.iterations = {self.smooth_iterations}
smooth.lambda_factor = 0.5
bpy.ops.object.modifier_apply(modifier="Smooth")
print(f"Applied smoothing")
'''

        code += f'''
# 6. Post-refinement analysis
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
post_non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
bpy.ops.object.mode_set(mode='OBJECT')
print(f"Post-refinement non-manifold edges: {{post_non_manifold}}")
'''

        if scale_for_comsol:
            code += f'''
# 7. Scale for COMSOL (mm to m)
obj.scale = ({scale}, {scale}, {scale})
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
print("Scaled mesh from mm to m for COMSOL")
'''

        code += f'''
# 8. Export refined mesh
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.ops.wm.stl_export(
    filepath=r"{output_path}",
    export_selected_objects=True,
    ascii_format=False
)

print(f"\\n===== REFINEMENT COMPLETE =====")
print(f"Output: {output_path}")
print(f"Final mesh: {{len(obj.data.vertices)}} vertices, {{len(obj.data.polygons)}} faces")
print(f"Non-manifold edges: {{pre_non_manifold}} -> {{post_non_manifold}}")
'''
        return code.strip()


def analyze_mesh_with_trimesh(mesh_path: Union[str, Path]) -> MeshQualityReport:
    """
    Analyze mesh quality using trimesh (fallback when Blender is not available).

    Args:
        mesh_path: Path to mesh file

    Returns:
        MeshQualityReport with analysis results
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for mesh analysis")

    mesh = trimesh.load(mesh_path)

    # Check manifold status
    is_watertight = mesh.is_watertight
    is_volume = mesh.is_volume

    # Get bounds
    bounds = mesh.bounds.tolist() if mesh.bounds is not None else [[0, 0, 0], [0, 0, 0]]

    return MeshQualityReport(
        is_manifold=is_volume,
        is_watertight=is_watertight,
        non_manifold_edges=0 if is_volume else -1,  # trimesh doesn't expose this directly
        non_manifold_verts=0 if is_volume else -1,
        degenerate_faces=len(mesh.faces) - len(mesh.nondegenerate_faces()) if hasattr(mesh, 'nondegenerate_faces') else 0,
        isolated_verts=0,
        volume=float(mesh.volume) if is_watertight else None,
        surface_area=float(mesh.area),
        bounding_box=bounds,
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
    )


def refine_mesh_with_trimesh(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    simplify_ratio: float = 0.5,
    smooth: bool = True,
) -> Path:
    """
    Refine mesh using trimesh (fallback when Blender is not available).

    Note: trimesh cannot perform Voxel Remesh like Blender.
    This provides basic repair and simplification only.

    Args:
        input_path: Input mesh path
        output_path: Output mesh path
        simplify_ratio: Target ratio for simplification
        smooth: Apply smoothing

    Returns:
        Path to output file
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for mesh refinement")

    mesh = trimesh.load(input_path)

    # Fill holes
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    # Fix normals
    mesh.fix_normals()

    # Merge vertices
    mesh.merge_vertices()

    # Simplify if needed
    if simplify_ratio < 1.0:
        try:
            target_faces = int(len(mesh.faces) * simplify_ratio)
            mesh = mesh.simplify_quadric_decimation(target_faces)
        except Exception as e:
            logger.warning(f"Simplification failed: {e}")

    # Smooth
    if smooth:
        trimesh.smoothing.filter_laplacian(mesh, iterations=3)

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))

    logger.info(f"Refined mesh saved to {output_path}")
    return output_path
