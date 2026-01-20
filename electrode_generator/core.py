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
