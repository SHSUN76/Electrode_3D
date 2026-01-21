"""
Tests for postprocessing module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from electrode_generator.config import PostprocessingConfig

# Check if skimage is available
try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Check if trimesh is available
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


@pytest.mark.skipif(not SKIMAGE_AVAILABLE, reason="scikit-image not installed")
class TestVoxelToMesh:
    """Tests for VoxelToMesh class."""

    @pytest.fixture
    def converter(self):
        """Create converter instance."""
        from postprocessing.mesh_converter import VoxelToMesh

        config = PostprocessingConfig(voxel_size=1.0, level=0.5)
        return VoxelToMesh(config)

    @pytest.fixture
    def sample_voxels(self):
        """Create sample voxel data with a sphere."""
        size = 32
        x, y, z = np.ogrid[:size, :size, :size]
        center = size // 2
        radius = size // 4
        sphere = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2 <= radius ** 2
        return sphere.astype(np.uint8)

    def test_convert_produces_vertices_and_faces(self, converter, sample_voxels):
        """Test that conversion produces vertices and faces."""
        verts, faces = converter.convert(sample_voxels, phase_id=1)

        assert len(verts) > 0
        assert len(faces) > 0
        assert verts.shape[1] == 3  # 3D coordinates
        assert faces.shape[1] == 3  # Triangular faces

    def test_convert_empty_volume(self, converter):
        """Test conversion of empty volume."""
        empty_voxels = np.zeros((32, 32, 32), dtype=np.uint8)

        with pytest.raises(Exception):
            # Marching cubes should fail or return empty on empty volume
            converter.convert(empty_voxels, phase_id=1)

    def test_convert_multiphase(self, converter):
        """Test multi-phase conversion."""
        # Create volume with multiple phases
        voxels = np.zeros((32, 32, 32), dtype=np.uint8)
        voxels[5:15, 5:15, 5:15] = 1
        voxels[18:28, 18:28, 18:28] = 2

        meshes = converter.convert_multiphase(voxels)

        assert 1 in meshes
        assert 2 in meshes
        assert 0 not in meshes  # Background not included


@pytest.mark.skipif(not TRIMESH_AVAILABLE or not SKIMAGE_AVAILABLE,
                    reason="trimesh or scikit-image not installed")
class TestMeshRefinement:
    """Tests for MeshRefinement class."""

    @pytest.fixture
    def refinement(self):
        """Create refinement instance."""
        from postprocessing.mesh_converter import MeshRefinement

        config = PostprocessingConfig(smooth_iterations=3, smooth_factor=0.5)
        return MeshRefinement(config)

    @pytest.fixture
    def sample_mesh(self):
        """Create sample mesh (cube)."""
        return trimesh.creation.box(extents=[1, 1, 1])

    def test_smooth(self, refinement, sample_mesh):
        """Test mesh smoothing."""
        original_verts = sample_mesh.vertices.copy()
        smoothed = refinement.smooth(sample_mesh.copy(), iterations=5)

        # Smoothed mesh should have same number of vertices
        assert len(smoothed.vertices) == len(original_verts)

    def test_simplify(self, refinement):
        """Test mesh simplification."""
        # Create dense sphere
        sphere = trimesh.creation.icosphere(subdivisions=4)
        original_faces = len(sphere.faces)

        simplified = refinement.simplify(sphere, ratio=0.5)

        # Should have fewer faces
        assert len(simplified.faces) < original_faces

    def test_repair(self, refinement, sample_mesh):
        """Test mesh repair."""
        repaired = refinement.repair(sample_mesh.copy())

        # Repaired mesh should be valid
        assert repaired.is_volume

    def test_validate(self, refinement, sample_mesh):
        """Test mesh validation."""
        validation = refinement.validate(sample_mesh)

        assert "is_watertight" in validation
        assert "is_volume" in validation
        assert "num_vertices" in validation
        assert "num_faces" in validation
        assert "surface_area" in validation


@pytest.mark.skipif(not SKIMAGE_AVAILABLE, reason="scikit-image not installed")
class TestTPMSGenerator:
    """Tests for TPMS structure generation."""

    @pytest.fixture
    def generator(self):
        """Create TPMS generator instance."""
        from postprocessing.mesh_converter import TPMSGenerator
        return TPMSGenerator()

    def test_generate_gyroid(self, generator):
        """Test gyroid generation."""
        verts, faces = generator.generate(
            tpms_type="gyroid",
            resolution=30,
            periods=1,
            thickness=0.3,
            dimensions=(5.0, 5.0, 5.0),
        )

        assert len(verts) > 0
        assert len(faces) > 0

    def test_generate_schwarz_p(self, generator):
        """Test Schwarz P generation."""
        verts, faces = generator.generate(
            tpms_type="schwarz_p",
            resolution=30,
            periods=1,
        )

        assert len(verts) > 0
        assert len(faces) > 0

    def test_generate_invalid_type(self, generator):
        """Test invalid TPMS type raises error."""
        with pytest.raises(ValueError, match="Unknown TPMS type"):
            generator.generate(tpms_type="invalid_type")

    @pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not installed")
    def test_generate_mesh(self, generator):
        """Test mesh generation."""
        mesh = generator.generate_mesh(
            tpms_type="gyroid",
            resolution=20,
        )

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not installed")
class TestMeshExporter:
    """Tests for MeshExporter class."""

    @pytest.fixture
    def exporter(self):
        """Create exporter instance."""
        from postprocessing.export import MeshExporter
        return MeshExporter()

    @pytest.fixture
    def sample_mesh(self):
        """Create sample mesh."""
        return trimesh.creation.box(extents=[1, 1, 1])

    def test_export_stl(self, exporter, sample_mesh):
        """Test STL export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.stl"
            result = exporter.export(sample_mesh, path, format="stl")

            assert result.exists()
            assert result.suffix == ".stl"

    def test_export_obj(self, exporter, sample_mesh):
        """Test OBJ export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.obj"
            result = exporter.export(sample_mesh, path, format="obj")

            assert result.exists()
            assert result.suffix == ".obj"

    def test_export_stl_ascii(self, exporter):
        """Test ASCII STL export."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
        ], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ascii.stl"
            result = exporter.export_stl_ascii(vertices, faces, path, name="tetrahedron")

            assert result.exists()

            # Check content
            content = result.read_text()
            assert "solid tetrahedron" in content
            assert "endsolid tetrahedron" in content
            assert "facet normal" in content

    def test_export_obj_manual(self, exporter):
        """Test manual OBJ export."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.obj"
            result = exporter.export_obj(vertices, faces, path)

            assert result.exists()

            content = result.read_text()
            assert "v " in content
            assert "f " in content

    def test_export_creates_directory(self, exporter, sample_mesh):
        """Test that export creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir1" / "subdir2" / "test.stl"
            result = exporter.export(sample_mesh, path)

            assert result.exists()
            assert result.parent.exists()

    def test_export_batch(self, exporter, sample_mesh):
        """Test batch export."""
        meshes = {
            "mesh1": sample_mesh,
            "mesh2": trimesh.creation.icosphere(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results = exporter.export_batch(
                meshes,
                tmpdir,
                formats=["stl", "obj"],
            )

            assert "mesh1" in results
            assert "mesh2" in results
            assert len(results["mesh1"]) == 2
            assert len(results["mesh2"]) == 2

    def test_export_nastran(self, exporter):
        """Test NASTRAN format export."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
        ], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.nas"
            result = exporter.export_nastran(vertices, faces, path, title="Test Mesh")

            assert result.exists()
            assert result.suffix == ".nas"

            content = result.read_text()
            assert "BEGIN BULK" in content
            assert "GRID*" in content
            assert "CTRIA3" in content
            assert "ENDDATA" in content

    def test_export_gmsh(self, exporter):
        """Test Gmsh MSH format export."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.msh"
            result = exporter.export_gmsh(vertices, faces, path)

            assert result.exists()
            assert result.suffix == ".msh"

            content = result.read_text()
            assert "$MeshFormat" in content
            assert "$Nodes" in content
            assert "$Elements" in content

    def test_export_for_comsol_nastran(self, exporter, sample_mesh):
        """Test COMSOL export with NASTRAN format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comsol_test.nas"
            result = exporter.export_for_comsol(
                sample_mesh,
                path,
                scale=0.001,  # mm to m
                format="nastran"
            )

            assert result.exists()
            assert result.suffix == ".nas"


class TestBlenderIntegration:
    """Tests for Blender MCP integration."""

    @pytest.fixture
    def refiner(self):
        """Create BlenderMeshRefiner instance."""
        from postprocessing.blender_integration import BlenderMeshRefiner
        return BlenderMeshRefiner(voxel_size=2.0, smooth_iterations=3)

    def test_get_import_code(self, refiner):
        """Test generating Blender import code."""
        code = refiner.get_import_code("test.stl", name="electrode")

        assert "import bpy" in code
        assert "stl_import" in code
        assert "electrode" in code

    def test_get_analyze_code(self, refiner):
        """Test generating Blender analysis code."""
        code = refiner.get_analyze_code("electrode")

        assert "import bpy" in code
        assert "import bmesh" in code
        assert "non_manifold" in code
        assert "MESH_QUALITY_RESULT" in code

    def test_get_remesh_code(self, refiner):
        """Test generating Blender remesh code."""
        code = refiner.get_remesh_code("electrode", voxel_size=1.5, smooth=True)

        assert "REMESH" in code
        assert "VOXEL" in code
        assert "1.5" in code
        assert "LAPLACIANSMOOTH" in code

    def test_get_export_code(self, refiner):
        """Test generating Blender export code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.stl"
            code = refiner.get_export_code(output, object_name="electrode", format="stl")

            assert "stl_export" in code
            assert "electrode" in code

    def test_get_full_refinement_code(self, refiner):
        """Test generating complete refinement pipeline code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.stl"
            output_path = Path(tmpdir) / "output.stl"

            code = refiner.get_full_refinement_code(
                input_path=input_path,
                output_path=output_path,
                voxel_size=2.0,
                smooth=True,
                scale_for_comsol=True,
            )

            assert "FULL MESH REFINEMENT PIPELINE" in code
            assert "stl_import" in code
            assert "REMESH" in code
            assert "LAPLACIANSMOOTH" in code
            assert "stl_export" in code
            assert "0.001" in code  # Scale factor


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not installed")
class TestMeshQualityAnalysis:
    """Tests for mesh quality analysis."""

    def test_analyze_mesh_with_trimesh(self):
        """Test mesh analysis with trimesh fallback."""
        from postprocessing.blender_integration import analyze_mesh_with_trimesh

        # Create a simple watertight mesh
        mesh = trimesh.creation.box(extents=[1, 1, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_path = Path(tmpdir) / "test.stl"
            mesh.export(str(mesh_path))

            report = analyze_mesh_with_trimesh(mesh_path)

            assert report.is_watertight
            assert report.vertex_count > 0
            assert report.face_count > 0
            assert report.surface_area > 0

    def test_mesh_quality_report_properties(self):
        """Test MeshQualityReport properties."""
        from postprocessing.blender_integration import MeshQualityReport

        report = MeshQualityReport(
            is_manifold=True,
            is_watertight=True,
            non_manifold_edges=0,
            non_manifold_verts=0,
            degenerate_faces=0,
            isolated_verts=0,
            volume=1.0,
            surface_area=6.0,
            bounding_box=[[0, 0, 0], [1, 1, 1]],
            vertex_count=8,
            face_count=12,
        )

        assert report.is_simulation_ready
        assert report.to_dict()["is_simulation_ready"]

    def test_refine_mesh_with_trimesh(self):
        """Test mesh refinement with trimesh fallback."""
        from postprocessing.blender_integration import refine_mesh_with_trimesh

        # Create mesh
        mesh = trimesh.creation.icosphere(subdivisions=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.stl"
            output_path = Path(tmpdir) / "output.stl"

            mesh.export(str(input_path))

            result = refine_mesh_with_trimesh(
                input_path,
                output_path,
                simplify_ratio=0.5,
                smooth=True,
            )

            assert result.exists()

            # Load refined mesh
            refined = trimesh.load(result)
            assert len(refined.faces) > 0
