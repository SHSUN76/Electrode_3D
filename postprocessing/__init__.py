"""
Postprocessing module for Electrode 3D Generator.

Contains:
- Voxel to mesh conversion (Marching Cubes)
- Mesh refinement and repair
- STL/OBJ/NASTRAN/Gmsh export
- COMSOL integration
- Blender MCP integration for advanced mesh refinement
"""

from postprocessing.mesh_converter import VoxelToMesh, MeshRefinement, TPMSGenerator
from postprocessing.export import MeshExporter
from postprocessing.blender_integration import (
    BlenderMeshRefiner,
    MeshQualityReport,
    analyze_mesh_with_trimesh,
    refine_mesh_with_trimesh,
)

__all__ = [
    "VoxelToMesh",
    "MeshRefinement",
    "TPMSGenerator",
    "MeshExporter",
    "BlenderMeshRefiner",
    "MeshQualityReport",
    "analyze_mesh_with_trimesh",
    "refine_mesh_with_trimesh",
]
