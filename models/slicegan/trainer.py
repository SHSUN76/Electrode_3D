"""
SliceGAN Trainer class.

Handles the complete training pipeline including:
- Data preprocessing
- Training loop with WGAN-GP
- Checkpointing
- Generation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from electrode_generator.config import SliceGANConfig
from models.slicegan.generator import Generator3D
from models.slicegan.discriminator import Critic2D, WGANGPLoss

logger = logging.getLogger(__name__)


class SliceGAN:
    """
    SliceGAN training and generation manager.

    Orchestrates the training process:
    1. Generate 3D volumes using Generator
    2. Slice volumes along x, y, z axes
    3. Train 3 Discriminators (one per axis)
    4. Use WGAN-GP for stable training

    Args:
        config: SliceGAN configuration
        device: Computation device ("cuda" or "cpu")
    """

    def __init__(
        self,
        config: Optional[SliceGANConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or SliceGANConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.generator = Generator3D(
            nz=self.config.nz,
            ngf=self.config.ngf,
            nc=self.config.nc,
            imtype=self.config.imtype,
        ).to(self.device)

        # Three discriminators for x, y, z slices
        self.critics = nn.ModuleList([
            Critic2D(nc=self.config.nc, ndf=self.config.ndf).to(self.device)
            for _ in range(3)
        ])

        # Loss function
        self.loss_fn = WGANGPLoss(lambda_gp=self.config.lambda_gp)

        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr_g,
            betas=(self.config.beta1, self.config.beta2),
        )

        self.optimizer_ds = [
            optim.Adam(
                critic.parameters(),
                lr=self.config.lr_d,
                betas=(self.config.beta1, self.config.beta2),
            )
            for critic in self.critics
        ]

        logger.info(f"SliceGAN initialized on device: {self.device}")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess training image and create patches.

        Args:
            image: Input image as numpy array

        Returns:
            Tensor of patches ready for training
        """
        img_size = self.config.img_size

        # Handle different image types
        if self.config.imtype == "nphase":
            # Segment into phases
            unique_vals = np.unique(image)
            nc = len(unique_vals)

            if len(image.shape) == 2:
                h, w = image.shape
                segmented = np.zeros((nc, h, w), dtype=np.float32)
                for i, val in enumerate(unique_vals):
                    segmented[i] = (image == val).astype(np.float32)
            else:
                h, w = image.shape[:2]
                segmented = np.zeros((nc, h, w), dtype=np.float32)
                for i, val in enumerate(unique_vals):
                    if len(image.shape) == 3:
                        mask = np.all(image == val, axis=-1)
                    else:
                        mask = image == val
                    segmented[i] = mask.astype(np.float32)

            image = segmented
        else:
            # Grayscale or color
            if len(image.shape) == 2:
                image = image[np.newaxis, ...]
            elif len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
                image = image.transpose(2, 0, 1)
            image = image.astype(np.float32) / image.max()

        # Create random patches
        nc, h, w = image.shape
        num_patches = 900  # Default number of patches

        patches = []
        for _ in range(num_patches):
            x = np.random.randint(0, w - img_size)
            y = np.random.randint(0, h - img_size)
            patch = image[:, y:y + img_size, x:x + img_size]
            patches.append(patch)

        patches = np.array(patches)
        return torch.FloatTensor(patches)

    def slice_volume(
        self,
        volume: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """
        Slice 3D volume along specified axis.

        Args:
            volume: 3D volume (batch, C, D, H, W)
            axis: Axis to slice (0=x, 1=y, 2=z)

        Returns:
            2D slice (batch, C, H, W)
        """
        batch_size = volume.size(0)
        depth = volume.size(2 + axis)

        # Random slice index
        idx = torch.randint(0, depth, (1,)).item()

        if axis == 0:  # X-axis
            return volume[:, :, idx, :, :]
        elif axis == 1:  # Y-axis
            return volume[:, :, :, idx, :]
        else:  # Z-axis
            return volume[:, :, :, :, idx]

    def train_step(
        self,
        real_data: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            real_data: Batch of real 2D patches

        Returns:
            Dictionary of loss values
        """
        batch_size = real_data.size(0)

        # ========================================
        # Train Critics
        # ========================================
        critic_losses = []
        wasserstein_dists = []
        gp_losses = []

        for _ in range(self.config.critic_iters):
            for opt in self.optimizer_ds:
                opt.zero_grad()

            # Generate 3D volume
            noise = torch.randn(
                batch_size, self.config.nz, 4, 4, 4,
                device=self.device
            )
            fake_volume = self.generator(noise)

            # Train each critic
            for axis, (critic, opt) in enumerate(zip(self.critics, self.optimizer_ds)):
                # Get fake slice
                fake_slice = self.slice_volume(fake_volume, axis)

                # Calculate critic loss
                loss, w_dist, gp = self.loss_fn.critic_loss(
                    critic, real_data, fake_slice, self.device
                )

                loss.backward(retain_graph=(axis < 2))
                opt.step()

                critic_losses.append(loss.item())
                wasserstein_dists.append(w_dist)
                gp_losses.append(gp)

        # ========================================
        # Train Generator
        # ========================================
        self.optimizer_g.zero_grad()

        noise = torch.randn(
            batch_size, self.config.nz, 4, 4, 4,
            device=self.device
        )
        fake_volume = self.generator(noise)

        gen_loss = 0
        for axis, critic in enumerate(self.critics):
            fake_slice = self.slice_volume(fake_volume, axis)
            gen_loss += self.loss_fn.generator_loss(critic, fake_slice)

        gen_loss = gen_loss / 3  # Average across axes
        gen_loss.backward()
        self.optimizer_g.step()

        return {
            "d_loss": np.mean(critic_losses),
            "g_loss": gen_loss.item(),
            "wasserstein": np.mean(wasserstein_dists),
            "gp": np.mean(gp_losses),
        }

    def train(
        self,
        image: np.ndarray,
        epochs: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
        save_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train SliceGAN on a 2D image.

        Args:
            image: Training image as numpy array
            epochs: Number of epochs (uses config if None)
            save_dir: Directory for checkpoints
            save_interval: Save every N epochs

        Returns:
            Training history
        """
        epochs = epochs or self.config.num_epochs

        # Preprocess and create dataloader
        patches = self.preprocess_image(image)
        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Training history
        history = {
            "d_loss": [],
            "g_loss": [],
            "wasserstein": [],
            "gp": [],
        }

        # Training loop
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            epoch_losses = {k: [] for k in history}

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                real_data = batch[0].to(self.device)

                # Training step
                losses = self.train_step(real_data)

                # Record losses
                for k, v in losses.items():
                    epoch_losses[k].append(v)

                # Update progress bar
                pbar.set_postfix({
                    "D": f"{losses['d_loss']:.4f}",
                    "G": f"{losses['g_loss']:.4f}",
                    "W": f"{losses['wasserstein']:.4f}",
                })

            # Average epoch losses
            for k in history:
                avg = np.mean(epoch_losses[k])
                history[k].append(avg)

            logger.info(
                f"Epoch {epoch + 1}: D_loss={history['d_loss'][-1]:.4f}, "
                f"G_loss={history['g_loss'][-1]:.4f}"
            )

            # Save checkpoint
            if save_dir and (epoch + 1) % save_interval == 0:
                self.save(Path(save_dir) / f"checkpoint_epoch_{epoch + 1}.pt")

        # Save final checkpoint
        if save_dir:
            self.save(Path(save_dir) / "latest.pt")

        return history

    def generate(
        self,
        size: Tuple[int, int, int] = (64, 64, 64),
        num_samples: int = 1,
        seed: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Generate 3D structures.

        Args:
            size: Output size (D, H, W)
            num_samples: Number of structures to generate
            seed: Random seed

        Returns:
            List of generated 3D arrays
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.generator.eval()

        with torch.no_grad():
            results = []

            for _ in range(num_samples):
                noise = torch.randn(
                    1, self.config.nz, 4, 4, 4,
                    device=self.device
                )
                volume = self.generator(noise)

                # Interpolate if different size
                if size != (64, 64, 64):
                    volume = torch.nn.functional.interpolate(
                        volume, size=size, mode="trilinear", align_corners=False
                    )

                # Convert to numpy
                volume = volume.squeeze(0).cpu().numpy()

                # Convert from one-hot to labels if nphase
                if self.config.imtype == "nphase":
                    volume = np.argmax(volume, axis=0)

                results.append(volume)

        self.generator.train()
        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config,
            "generator_state_dict": self.generator.state_dict(),
            "critics_state_dict": [c.state_dict() for c in self.critics],
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_ds_state_dict": [o.state_dict() for o in self.optimizer_ds],
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to: {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])

        for critic, state_dict in zip(self.critics, checkpoint["critics_state_dict"]):
            critic.load_state_dict(state_dict)

        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])

        for opt, state_dict in zip(self.optimizer_ds, checkpoint["optimizer_ds_state_dict"]):
            opt.load_state_dict(state_dict)

        logger.info(f"Checkpoint loaded from: {path}")
