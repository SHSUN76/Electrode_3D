"""
Visualization utilities for electrode microstructures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None
    cm = None

try:
    import pyvista as pv
except ImportError:
    pv = None


class Visualizer:
    """
    Visualization tools for voxel data and meshes.

    Supports:
    - 2D slice visualization
    - 3D volume rendering
    - Mesh visualization
    - Metrics plots
    """

    # Default colormap for electrode phases
    PHASE_COLORS = {
        0: [1.0, 1.0, 1.0, 0.1],    # Pore (transparent white)
        1: [0.2, 0.4, 0.8, 1.0],    # Active material (blue)
        2: [0.8, 0.4, 0.2, 1.0],    # Binder (orange)
        3: [0.2, 0.8, 0.2, 1.0],    # Conductive additive (green)
    }

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize

        if plt is None:
            raise ImportError("matplotlib is required. Install with: pip install matplotlib")

    def plot_slice(
        self,
        voxels: np.ndarray,
        axis: int = 2,
        index: Optional[int] = None,
        title: str = "Microstructure Slice",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot a 2D slice of the voxel volume.

        Args:
            voxels: 3D voxel array
            axis: Axis to slice (0=x, 1=y, 2=z)
            index: Slice index (middle if None)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
        """
        if index is None:
            index = voxels.shape[axis] // 2

        if axis == 0:
            slice_2d = voxels[index, :, :]
        elif axis == 1:
            slice_2d = voxels[:, index, :]
        else:
            slice_2d = voxels[:, :, index]

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create colormap
        cmap = self._create_phase_colormap(int(voxels.max()) + 1)

        im = ax.imshow(slice_2d.T, origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_title(f"{title} (axis={axis}, index={index})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label="Phase")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_slices_grid(
        self,
        voxels: np.ndarray,
        num_slices: int = 9,
        axis: int = 2,
        title: str = "Microstructure Slices",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot multiple slices in a grid.

        Args:
            voxels: 3D voxel array
            num_slices: Number of slices to show
            axis: Axis to slice
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
        """
        n_cols = int(np.ceil(np.sqrt(num_slices)))
        n_rows = int(np.ceil(num_slices / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        axes = np.atleast_2d(axes).flatten()

        indices = np.linspace(0, voxels.shape[axis] - 1, num_slices, dtype=int)
        cmap = self._create_phase_colormap(int(voxels.max()) + 1)

        for i, (ax, idx) in enumerate(zip(axes, indices)):
            if axis == 0:
                slice_2d = voxels[idx, :, :]
            elif axis == 1:
                slice_2d = voxels[:, idx, :]
            else:
                slice_2d = voxels[:, :, idx]

            ax.imshow(slice_2d.T, origin="lower", cmap=cmap, interpolation="nearest")
            ax.set_title(f"Slice {idx}")
            ax.axis("off")

        # Hide unused axes
        for ax in axes[num_slices:]:
            ax.axis("off")

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_orthogonal_slices(
        self,
        voxels: np.ndarray,
        title: str = "Orthogonal Views",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot orthogonal slices (XY, XZ, YZ) at the center.

        Args:
            voxels: 3D voxel array
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        cmap = self._create_phase_colormap(int(voxels.max()) + 1)

        # XY plane (Z slice)
        z_mid = voxels.shape[2] // 2
        axes[0].imshow(voxels[:, :, z_mid].T, origin="lower", cmap=cmap)
        axes[0].set_title(f"XY Plane (Z={z_mid})")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        # XZ plane (Y slice)
        y_mid = voxels.shape[1] // 2
        axes[1].imshow(voxels[:, y_mid, :].T, origin="lower", cmap=cmap)
        axes[1].set_title(f"XZ Plane (Y={y_mid})")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")

        # YZ plane (X slice)
        x_mid = voxels.shape[0] // 2
        axes[2].imshow(voxels[x_mid, :, :].T, origin="lower", cmap=cmap)
        axes[2].set_title(f"YZ Plane (X={x_mid})")
        axes[2].set_xlabel("Y")
        axes[2].set_ylabel("Z")

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_volume_3d(
        self,
        voxels: np.ndarray,
        phase_id: int = 1,
        title: str = "3D Volume",
        opacity: float = 0.5,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot 3D volume rendering using PyVista.

        Args:
            voxels: 3D voxel array
            phase_id: Phase to render
            title: Plot title
            opacity: Volume opacity
            save_path: Path to save figure
            show: Whether to display
        """
        if pv is None:
            raise ImportError("pyvista is required for 3D visualization")

        # Create uniform grid
        grid = pv.ImageData()
        grid.dimensions = np.array(voxels.shape) + 1
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)

        # Add voxel data
        binary = (voxels == phase_id).astype(np.float32)
        grid.cell_data["values"] = binary.flatten(order="F")

        # Threshold to get surface
        threshed = grid.threshold(0.5)

        # Plot
        plotter = pv.Plotter(off_screen=not show)
        plotter.add_mesh(threshed, color=self.PHASE_COLORS.get(phase_id, [0.5, 0.5, 0.5])[:3], opacity=opacity)
        plotter.add_title(title)

        if save_path:
            plotter.screenshot(str(save_path))

        if show:
            plotter.show()
        else:
            plotter.close()

    def plot_metrics_bar(
        self,
        metrics: Dict[str, float],
        title: str = "Microstructure Metrics",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot metrics as bar chart.

        Args:
            metrics: Dictionary of metric names and values
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        keys = list(metrics.keys())
        values = list(metrics.values())

        # Filter out inf values
        valid_mask = [not np.isinf(v) for v in values]
        keys = [k for k, m in zip(keys, valid_mask) if m]
        values = [v for v, m in zip(values, valid_mask) if m]

        bars = ax.bar(keys, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(keys))))

        ax.set_ylabel("Value")
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_comparison(
        self,
        reference_metrics: Dict[str, float],
        generated_metrics: Dict[str, float],
        title: str = "Metrics Comparison",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot comparison between reference and generated metrics.

        Args:
            reference_metrics: Reference structure metrics
            generated_metrics: Generated structure metrics
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
        """
        # Get common keys
        keys = [k for k in reference_metrics if k in generated_metrics]

        ref_vals = [reference_metrics[k] for k in keys]
        gen_vals = [generated_metrics[k] for k in keys]

        # Filter inf values
        valid_mask = [not (np.isinf(r) or np.isinf(g)) for r, g in zip(ref_vals, gen_vals)]
        keys = [k for k, m in zip(keys, valid_mask) if m]
        ref_vals = [v for v, m in zip(ref_vals, valid_mask) if m]
        gen_vals = [v for v, m in zip(gen_vals, valid_mask) if m]

        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(len(keys))
        width = 0.35

        bars1 = ax.bar(x - width / 2, ref_vals, width, label="Reference", color="steelblue")
        bars2 = ax.bar(x + width / 2, gen_vals, width, label="Generated", color="darkorange")

        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot training loss history.

        Args:
            history: Dictionary with loss names and values per epoch
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for name, values in history.items():
            ax.plot(values, label=name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_particle_size_distribution(
        self,
        bin_edges: np.ndarray,
        counts: np.ndarray,
        title: str = "Particle Size Distribution",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        Plot particle size distribution histogram.

        Args:
            bin_edges: Histogram bin edges
            counts: Histogram counts
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
        """
        if len(bin_edges) == 0:
            print("No data to plot")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)

        ax.bar(bin_centers, counts, width=bin_widths * 0.9, color="steelblue", edgecolor="black")

        ax.set_xlabel("Particle Diameter (μm)")
        ax.set_ylabel("Count")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def _create_phase_colormap(self, num_phases: int) -> ListedColormap:
        """Create colormap for phase visualization."""
        colors = []
        for i in range(num_phases):
            if i in self.PHASE_COLORS:
                colors.append(self.PHASE_COLORS[i][:3])
            else:
                # Generate random color for unknown phases
                colors.append(plt.cm.tab10(i % 10)[:3])

        return ListedColormap(colors)


def save_animation_gif(
    voxels: np.ndarray,
    output_path: Union[str, Path],
    axis: int = 2,
    fps: int = 10,
) -> None:
    """
    Save animated GIF of slices through volume.

    Args:
        voxels: 3D voxel array
        output_path: Path for output GIF
        axis: Axis to animate through
        fps: Frames per second
    """
    if plt is None:
        raise ImportError("matplotlib is required")

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        raise ImportError("Pillow is required for GIF export")

    fig, ax = plt.subplots(figsize=(6, 6))

    # Create colormap
    cmap = ListedColormap([[1, 1, 1], [0.2, 0.4, 0.8], [0.8, 0.4, 0.2], [0.2, 0.8, 0.2]])

    # Initial frame
    if axis == 0:
        im = ax.imshow(voxels[0, :, :].T, origin="lower", cmap=cmap, animated=True)
    elif axis == 1:
        im = ax.imshow(voxels[:, 0, :].T, origin="lower", cmap=cmap, animated=True)
    else:
        im = ax.imshow(voxels[:, :, 0].T, origin="lower", cmap=cmap, animated=True)

    ax.axis("off")

    def update(frame):
        if axis == 0:
            im.set_data(voxels[frame, :, :].T)
        elif axis == 1:
            im.set_data(voxels[:, frame, :].T)
        else:
            im.set_data(voxels[:, :, frame].T)
        return [im]

    num_frames = voxels.shape[axis]
    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 // fps, blit=True)

    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close()
