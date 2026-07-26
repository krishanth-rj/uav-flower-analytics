"""Matplotlib visualizations for already-computed UAV flower analytics data.

This module is intentionally limited to rendering supplied density matrices;
it does not calculate density, load models, or read image files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_density_grid(
    density_grid: np.ndarray,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Render a flower-density matrix as a heatmap.

    Args:
        density_grid: Two-dimensional NumPy array of flower counts or density
            values, indexed as ``[grid_row, grid_column]``.
        save_path: Optional destination for a high-resolution figure image.
            Missing parent directories are created automatically.
        show: Whether to display the figure through Matplotlib's active backend.

    Returns:
        The created Matplotlib figure and its primary axes.

    Raises:
        ValueError: If ``density_grid`` is not two-dimensional.
    """
    if density_grid.ndim != 2:
        raise ValueError("density_grid must be a two-dimensional NumPy array.")

    fig, ax = plt.subplots()
    # imshow maps each matrix element directly to its corresponding grid cell.
    image = ax.imshow(density_grid, cmap="viridis", interpolation="nearest")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Flower Density")
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    ax.set_title("Flower Density Heatmap")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    return fig, ax
