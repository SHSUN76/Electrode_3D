"""
Configuration management for Electrode 3D Generator.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class SliceGANConfig:
    """SliceGAN model configuration."""

    # Network architecture
    nz: int = 512  # Latent vector dimension
    ngf: int = 64  # Generator feature maps
    ndf: int = 64  # Discriminator feature maps
    nc: int = 3    # Number of channels (phases)

    # Training parameters
    batch_size: int = 8
    lr_g: float = 0.0001
    lr_d: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.99
    lambda_gp: float = 10.0
    critic_iters: int = 5
    num_epochs: int = 100

    # Data parameters
    img_size: int = 64
    scale_factor: float = 1.0
    imtype: str = "nphase"  # "nphase", "grayscale", "color"


@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""

    # Image processing
    target_size: tuple = (256, 256)
    normalize: bool = True
    denoise: bool = True
    denoise_sigma: float = 1.0

    # Segmentation
    segmentation_model: str = "unet3d"  # "unet3d", "swin_unetr"
    num_classes: int = 3  # Active material, pore, binder

    # Augmentation
    augment: bool = True
    rotation_range: float = 90.0
    flip_horizontal: bool = True
    flip_vertical: bool = True


@dataclass
class PostprocessingConfig:
    """Postprocessing pipeline configuration."""

    # Marching cubes
    voxel_size: float = 0.1  # micrometers
    level: float = 0.5
    step_size: int = 1

    # Mesh refinement
    remesh: bool = True
    remesh_voxel_size: float = 0.05
    smooth_iterations: int = 2
    smooth_factor: float = 0.5

    # Export
    export_formats: List[str] = field(default_factory=lambda: ["stl", "obj"])
    export_scale: float = 1.0  # mm to target unit
    triangulate: bool = True


@dataclass
class COMSOLConfig:
    """COMSOL integration configuration."""

    # Connection
    cores: int = 4

    # Physics
    physics_type: str = "porous_electrode"

    # Material properties
    porosity: float = 0.3
    tortuosity: float = 1.5
    sigma_solid: float = 100.0  # S/m
    kappa_electrolyte: float = 1.0  # S/m

    # Mesh
    mesh_size: str = "normal"  # "coarse", "normal", "fine", "finer"

    # Study
    study_type: str = "stationary"  # "stationary", "time_dependent"


@dataclass
class Config:
    """Main configuration class."""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    # Sub-configurations
    slicegan: SliceGANConfig = field(default_factory=SliceGANConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    comsol: COMSOLConfig = field(default_factory=COMSOLConfig)

    # Device
    device: str = "cuda"  # "cuda", "cpu"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()

        # Update paths
        if "project_root" in data:
            config.project_root = Path(data["project_root"])
        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])
        if "output_dir" in data:
            config.output_dir = Path(data["output_dir"])
        if "checkpoint_dir" in data:
            config.checkpoint_dir = Path(data["checkpoint_dir"])

        # Update sub-configs
        if "slicegan" in data:
            for k, v in data["slicegan"].items():
                if hasattr(config.slicegan, k):
                    setattr(config.slicegan, k, v)

        if "preprocessing" in data:
            for k, v in data["preprocessing"].items():
                if hasattr(config.preprocessing, k):
                    setattr(config.preprocessing, k, v)

        if "postprocessing" in data:
            for k, v in data["postprocessing"].items():
                if hasattr(config.postprocessing, k):
                    setattr(config.postprocessing, k, v)

        if "comsol" in data:
            for k, v in data["comsol"].items():
                if hasattr(config.comsol, k):
                    setattr(config.comsol, k, v)

        # Update device and seed
        if "device" in data:
            config.device = data["device"]
        if "seed" in data:
            config.seed = data["seed"]

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "device": self.device,
            "seed": self.seed,
            "slicegan": {
                "nz": self.slicegan.nz,
                "ngf": self.slicegan.ngf,
                "ndf": self.slicegan.ndf,
                "nc": self.slicegan.nc,
                "batch_size": self.slicegan.batch_size,
                "lr_g": self.slicegan.lr_g,
                "lr_d": self.slicegan.lr_d,
                "beta1": self.slicegan.beta1,
                "beta2": self.slicegan.beta2,
                "lambda_gp": self.slicegan.lambda_gp,
                "critic_iters": self.slicegan.critic_iters,
                "num_epochs": self.slicegan.num_epochs,
                "img_size": self.slicegan.img_size,
                "scale_factor": self.slicegan.scale_factor,
                "imtype": self.slicegan.imtype,
            },
            "preprocessing": {
                "target_size": list(self.preprocessing.target_size),
                "normalize": self.preprocessing.normalize,
                "denoise": self.preprocessing.denoise,
                "denoise_sigma": self.preprocessing.denoise_sigma,
                "segmentation_model": self.preprocessing.segmentation_model,
                "num_classes": self.preprocessing.num_classes,
                "augment": self.preprocessing.augment,
                "rotation_range": self.preprocessing.rotation_range,
                "flip_horizontal": self.preprocessing.flip_horizontal,
                "flip_vertical": self.preprocessing.flip_vertical,
            },
            "postprocessing": {
                "voxel_size": self.postprocessing.voxel_size,
                "level": self.postprocessing.level,
                "step_size": self.postprocessing.step_size,
                "remesh": self.postprocessing.remesh,
                "remesh_voxel_size": self.postprocessing.remesh_voxel_size,
                "smooth_iterations": self.postprocessing.smooth_iterations,
                "smooth_factor": self.postprocessing.smooth_factor,
                "export_formats": self.postprocessing.export_formats,
                "export_scale": self.postprocessing.export_scale,
                "triangulate": self.postprocessing.triangulate,
            },
            "comsol": {
                "cores": self.comsol.cores,
                "physics_type": self.comsol.physics_type,
                "porosity": self.comsol.porosity,
                "tortuosity": self.comsol.tortuosity,
                "sigma_solid": self.comsol.sigma_solid,
                "kappa_electrolyte": self.comsol.kappa_electrolyte,
                "mesh_size": self.comsol.mesh_size,
                "study_type": self.comsol.study_type,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        dirs = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "generated",
            self.output_dir,
            self.checkpoint_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
