"""
Core module for Electrode 3D Generator.

Provides the main ElectrodeGenerator class that orchestrates the entire pipeline:
1. Data preprocessing
2. SliceGAN training/inference
3. 3D structure generation
4. Mesh conversion
5. COMSOL export
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

from electrode_generator.config import Config

logger = logging.getLogger(__name__)


class ElectrodeGenerator:
    """
    Main class for generating 3D electrode microstructures.

    This class orchestrates the entire pipeline from 2D SEM images
    to 3D structures ready for COMSOL simulation.

    Example:
        >>> from electrode_generator import ElectrodeGenerator, Config
        >>> config = Config.from_yaml("configs/default.yaml")
        >>> generator = ElectrodeGenerator(config)
        >>> generator.train("data/raw/sem_image.tif")
        >>> structure = generator.generate(size=(64, 64, 64))
        >>> generator.export_mesh(structure, "output/electrode.stl")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the ElectrodeGenerator.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.config.ensure_dirs()

        self._model = None
        self._preprocessor = None
        self._mesh_converter = None

        logger.info(f"ElectrodeGenerator initialized with device: {self.config.device}")

    @property
    def model(self):
        """Lazy load SliceGAN model."""
        if self._model is None:
            from models.slicegan import SliceGAN
            self._model = SliceGAN(self.config.slicegan, device=self.config.device)
        return self._model

    @property
    def preprocessor(self):
        """Lazy load preprocessor."""
        if self._preprocessor is None:
            from preprocessing import ImagePreprocessor
            self._preprocessor = ImagePreprocessor(self.config.preprocessing)
        return self._preprocessor

    @property
    def mesh_converter(self):
        """Lazy load mesh converter."""
        if self._mesh_converter is None:
            from postprocessing import VoxelToMesh
            self._mesh_converter = VoxelToMesh(self.config.postprocessing)
        return self._mesh_converter

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess a 2D image.

        Args:
            path: Path to the image file (PNG, TIFF, etc.)

        Returns:
            Preprocessed image as numpy array
        """
        return self.preprocessor.load_and_preprocess(path)

    def train(
        self,
        image_path: Union[str, Path],
        epochs: Optional[int] = None,
        save_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train SliceGAN on a 2D image.

        Args:
            image_path: Path to training image
            epochs: Number of epochs (uses config if None)
            save_interval: Save checkpoint every N epochs

        Returns:
            Training history with loss values
        """
        logger.info(f"Starting training with image: {image_path}")

        # Load and preprocess image
        image = self.load_image(image_path)

        # Train model
        epochs = epochs or self.config.slicegan.num_epochs
        history = self.model.train(
            image,
            epochs=epochs,
            save_dir=self.config.checkpoint_dir,
            save_interval=save_interval,
        )

        logger.info("Training completed")
        return history

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load a trained model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        self.model.load(path)
        logger.info(f"Loaded checkpoint: {path}")

    def generate(
        self,
        size: Tuple[int, int, int] = (64, 64, 64),
        num_samples: int = 1,
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate 3D electrode structure(s).

        Args:
            size: Output volume size (D, H, W)
            num_samples: Number of structures to generate
            seed: Random seed for reproducibility

        Returns:
            Generated 3D volume(s) as numpy array(s)
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Generating {num_samples} structure(s) of size {size}")

        structures = self.model.generate(size=size, num_samples=num_samples)

        if num_samples == 1:
            return structures[0]
        return structures

    def voxel_to_mesh(
        self,
        voxels: np.ndarray,
        phase_id: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert voxel data to mesh.

        Args:
            voxels: 3D voxel array
            phase_id: Phase to extract (for multi-phase structures)

        Returns:
            Tuple of (vertices, faces) arrays
        """
        return self.mesh_converter.convert(voxels, phase_id=phase_id)

    def export_mesh(
        self,
        voxels: np.ndarray,
        output_path: Union[str, Path],
        phase_id: int = 1,
        format: str = "stl",
    ) -> Path:
        """
        Export voxel structure as mesh file.

        Args:
            voxels: 3D voxel array
            output_path: Output file path
            phase_id: Phase to export
            format: Export format ("stl", "obj")

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        vertices, faces = self.voxel_to_mesh(voxels, phase_id=phase_id)

        if format.lower() == "stl":
            self.mesh_converter.export_stl(vertices, faces, output_path)
        elif format.lower() == "obj":
            self.mesh_converter.export_obj(vertices, faces, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported mesh to: {output_path}")
        return output_path

    def calculate_metrics(self, voxels: np.ndarray) -> Dict[str, float]:
        """
        Calculate microstructure metrics.

        Args:
            voxels: 3D voxel array

        Returns:
            Dictionary of metrics (volume_fraction, surface_area, tortuosity, etc.)
        """
        from utils.metrics import MicrostructureMetrics
        calculator = MicrostructureMetrics(voxel_size=self.config.postprocessing.voxel_size)
        return calculator.calculate_all(voxels)

    def run_pipeline(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        train: bool = True,
        epochs: Optional[int] = None,
        num_structures: int = 1,
        structure_size: Tuple[int, int, int] = (64, 64, 64),
    ) -> Dict:
        """
        Run the complete pipeline.

        Args:
            image_path: Path to training image
            output_dir: Output directory
            train: Whether to train (False to use existing checkpoint)
            epochs: Training epochs
            num_structures: Number of structures to generate
            structure_size: Size of generated structures

        Returns:
            Dictionary with results and paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "input_image": str(image_path),
            "output_dir": str(output_dir),
            "structures": [],
            "meshes": [],
            "metrics": [],
        }

        # Train or load
        if train:
            history = self.train(image_path, epochs=epochs)
            results["training_history"] = history
        else:
            checkpoint = self.config.checkpoint_dir / "latest.pt"
            self.load_checkpoint(checkpoint)

        # Generate structures
        structures = self.generate(size=structure_size, num_samples=num_structures)
        if num_structures == 1:
            structures = [structures]

        for i, structure in enumerate(structures):
            # Save voxel data
            voxel_path = output_dir / f"structure_{i:04d}.npy"
            np.save(voxel_path, structure)
            results["structures"].append(str(voxel_path))

            # Export meshes
            for fmt in self.config.postprocessing.export_formats:
                mesh_path = output_dir / f"structure_{i:04d}.{fmt}"
                self.export_mesh(structure, mesh_path, format=fmt)
                results["meshes"].append(str(mesh_path))

            # Calculate metrics
            metrics = self.calculate_metrics(structure)
            results["metrics"].append(metrics)

        logger.info(f"Pipeline completed. Results saved to: {output_dir}")
        return results

    def load_microct_volume(
        self,
        folder_path: Union[str, Path],
        max_slices: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load micro-CT TIFF sequence as 3D volume.

        Args:
            folder_path: Path to folder containing TIFF slices
            max_slices: Maximum number of slices to load (None for all)

        Returns:
            3D numpy array (D, H, W)
        """
        from preprocessing import StackProcessor

        processor = StackProcessor(self.config.preprocessing)
        volume = processor.load_tiff_sequence(folder_path, max_slices=max_slices)

        logger.info(f"Loaded micro-CT volume: {volume.shape}")
        return volume

    def extract_training_slices(
        self,
        volume: np.ndarray,
        num_slices: Optional[int] = None,
        axis: str = 'z',
    ) -> List[np.ndarray]:
        """
        Extract 2D slices from 3D volume for SliceGAN training.

        Args:
            volume: 3D numpy array
            num_slices: Number of slices to extract
            axis: Axis to slice along ('x', 'y', or 'z')

        Returns:
            List of 2D numpy arrays
        """
        from preprocessing import StackProcessor

        processor = StackProcessor(self.config.preprocessing)
        slices = processor.extract_training_slices(
            volume,
            axis=axis,
            num_slices=num_slices,
        )

        logger.info(f"Extracted {len(slices)} training slices")
        return slices

    def run_pipeline_with_refinement(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        use_blender: bool = True,
        blender_voxel_size: float = 2.0,
        run_comsol: bool = False,
        train: bool = True,
        epochs: Optional[int] = None,
        num_structures: int = 1,
        structure_size: Tuple[int, int, int] = (64, 64, 64),
    ) -> Dict:
        """
        Run complete pipeline with Blender mesh refinement and optional COMSOL.

        This is the full workflow:
        1. Load and preprocess input (2D image or micro-CT volume)
        2. Train SliceGAN (or load existing model)
        3. Generate 3D structures
        4. Convert to mesh (Marching Cubes)
        5. Refine mesh with Blender MCP (Voxel Remesh)
        6. Export for COMSOL simulation
        7. Optionally run COMSOL simulation

        Args:
            input_path: Path to training image or micro-CT folder
            output_dir: Output directory
            use_blender: Use Blender MCP for mesh refinement
            blender_voxel_size: Voxel size for Blender remesh
            run_comsol: Run COMSOL simulation after mesh generation
            train: Whether to train (False to use existing checkpoint)
            epochs: Training epochs
            num_structures: Number of structures to generate
            structure_size: Size of generated structures

        Returns:
            Dictionary with results and paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "structures": [],
            "raw_meshes": [],
            "refined_meshes": [],
            "metrics": [],
            "blender_codes": [],
            "comsol_models": [],
        }

        input_path = Path(input_path)

        # Step 1: Handle micro-CT or single image input
        if input_path.is_dir():
            # Load micro-CT volume and extract training slices
            volume = self.load_microct_volume(input_path)
            slices = self.extract_training_slices(volume, num_slices=50)

            # Save slices for training
            slice_dir = output_dir / "training_slices"
            slice_dir.mkdir(exist_ok=True)
            for i, s in enumerate(slices):
                from PIL import Image
                img = Image.fromarray((s * 255 / s.max()).astype(np.uint8))
                img.save(slice_dir / f"slice_{i:04d}.png")

            # Use first slice as representative for training
            training_image = slices[0]
            results["microct_volume_shape"] = list(volume.shape)
        else:
            # Load single image
            training_image = self.load_image(input_path)

        # Step 2: Train or load model
        if train:
            logger.info("Training SliceGAN model...")
            # For micro-CT, we'd want to use multiple slices
            # Here we simplify by using a representative slice
            history = self.model.train(
                training_image,
                epochs=epochs or self.config.slicegan.num_epochs,
                save_dir=self.config.checkpoint_dir,
            )
            results["training_history"] = history
        else:
            checkpoint = self.config.checkpoint_dir / "latest.pt"
            self.load_checkpoint(checkpoint)

        # Step 3: Generate 3D structures
        logger.info(f"Generating {num_structures} structure(s)...")
        structures = self.generate(size=structure_size, num_samples=num_structures)
        if num_structures == 1:
            structures = [structures]

        # Step 4-6: Process each structure
        from postprocessing import BlenderMeshRefiner, MeshExporter

        blender_refiner = BlenderMeshRefiner(voxel_size=blender_voxel_size)
        exporter = MeshExporter(self.config.postprocessing)

        for i, structure in enumerate(structures):
            logger.info(f"Processing structure {i+1}/{num_structures}")

            # Save voxel data
            voxel_path = output_dir / f"structure_{i:04d}.npy"
            np.save(voxel_path, structure)
            results["structures"].append(str(voxel_path))

            # Convert to mesh (Marching Cubes)
            vertices, faces = self.voxel_to_mesh(structure)

            # Export raw mesh
            raw_mesh_path = output_dir / f"structure_{i:04d}_raw.stl"
            self.mesh_converter.export_stl(vertices, faces, raw_mesh_path)
            results["raw_meshes"].append(str(raw_mesh_path))

            # Step 5: Blender refinement
            if use_blender:
                refined_mesh_path = output_dir / f"structure_{i:04d}_refined.stl"

                # Generate Blender code for mesh refinement
                blender_code = blender_refiner.get_full_refinement_code(
                    input_path=raw_mesh_path,
                    output_path=refined_mesh_path,
                    voxel_size=blender_voxel_size,
                    scale_for_comsol=True,
                )
                results["blender_codes"].append(blender_code)

                # Save Blender code for manual execution or MCP
                code_path = output_dir / f"blender_refine_{i:04d}.py"
                with open(code_path, "w") as f:
                    f.write(blender_code)

                logger.info(f"Blender refinement code saved to: {code_path}")
                logger.info("Execute via Blender MCP: mcp__blender__execute_blender_code")

                results["refined_meshes"].append(str(refined_mesh_path))
            else:
                # Use trimesh fallback
                from postprocessing import refine_mesh_with_trimesh

                refined_mesh_path = output_dir / f"structure_{i:04d}_refined.stl"
                refine_mesh_with_trimesh(raw_mesh_path, refined_mesh_path)
                results["refined_meshes"].append(str(refined_mesh_path))

            # Calculate metrics
            metrics = self.calculate_metrics(structure)
            results["metrics"].append(metrics)

            # Step 6: Export for COMSOL
            comsol_mesh_path = output_dir / f"structure_{i:04d}_comsol.nas"

            import trimesh as tm
            mesh = tm.Trimesh(vertices=vertices, faces=faces)
            exporter.export_for_comsol(
                mesh,
                comsol_mesh_path,
                scale=1e-3,  # mm to m
                format="nastran"
            )
            logger.info(f"COMSOL mesh exported: {comsol_mesh_path}")

        # Step 7: COMSOL simulation (optional)
        if run_comsol:
            logger.info("COMSOL simulation requested...")
            try:
                from comsol.interface import COMSOLInterface, COMSOLModelBuilder

                with COMSOLInterface(self.config.comsol) as comsol:
                    builder = COMSOLModelBuilder(comsol)

                    for i, refined_path in enumerate(results["refined_meshes"]):
                        model = builder.create_electrode_model(
                            refined_path,
                            electrode_type="cathode"
                        )
                        builder.setup_electrochemistry()
                        builder.setup_study()

                        comsol_results = builder.run_and_export(
                            output_dir / f"comsol_results_{i:04d}"
                        )
                        results["comsol_models"].append(comsol_results)

            except ImportError:
                logger.warning("COMSOL integration (mph) not available")
            except Exception as e:
                logger.error(f"COMSOL simulation failed: {e}")

        logger.info(f"Pipeline with refinement completed. Results saved to: {output_dir}")
        return results

    def segment_volume(
        self,
        volume: np.ndarray,
        n_phases: int = 3,
        method: str = "improved",
    ) -> np.ndarray:
        """
        Segment 3D volume into multiple phases.

        Args:
            volume: 3D grayscale volume
            n_phases: Number of phases
            method: Segmentation method ("improved", "otsu", "kmeans")

        Returns:
            Segmented 3D volume with integer labels
        """
        if method == "improved":
            # Use improved segmentation with morphological operations
            segmented = np.zeros_like(volume, dtype=np.uint8)

            for z in range(volume.shape[0]):
                segmented[z] = self.preprocessor.segment_multiphase_improved(
                    volume[z],
                    n_phases=n_phases,
                )

            return segmented
        else:
            # Use basic segmentation
            from preprocessing import StackProcessor
            processor = StackProcessor(self.config.preprocessing)
            return processor.preprocess_stack(volume, segment=True)
