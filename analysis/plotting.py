import numpy as np
import matplotlib.pyplot as plt


def plot_free_path_distribution(free_paths: np.ndarray, sigma: float, bins: int = 50) -> None:
    if free_paths.size == 0:
        raise ValueError("free_paths must not be empty.")
    if sigma <= 0:
        raise ValueError("sigma must be a positive value.")
    if bins <= 0:
        raise ValueError("bins must be a positive integer.")
    if np.any(free_paths < 0):
        raise ValueError("free_paths must contain non-negative values.")
    x = np.linspace(0, free_paths.max(), 1000)
    theoretical_pdf = sigma * np.exp(-sigma * x)
    plt.figure(figsize=(10, 6))
    plt.hist(free_paths,
             bins=bins,
             density=True,
             alpha=0.6,
             label="Monte Carlo",
             edgecolor="black")
    plt.plot(x, theoretical_pdf, linewidth=2, label="Theoretical distribution")
    plt.xlabel("Длина свободного пробега")
    plt.ylabel("Плотность вероятности")
    plt.title("Распределение длины свободного пробега")
    theoretical_mean = 1 / sigma
    empirical_mean = np.mean(free_paths)
    plt.axvline(
        theoretical_mean,
        color="red",
        linestyle="--",
        label=f"Theoretical mean = {theoretical_mean:.4f}"
    )
    plt.axvline(
        empirical_mean,
        color="green",
        linestyle=":",
        label=f"Empirical mean = {empirical_mean:.4f}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_flux_slice(field: dict, radius: float):
    """
    Plot neutron flux in the XY plane (z = 0).
    """

    resolution = len(field["coordinates"])
    center = resolution // 2

    x = field["coordinates"]
    y = field["coordinates"]

    flux_slice = field["flux"][:, :, center].copy()
    mask_slice = field["mask"][:, :, center]

    flux_slice[~mask_slice] = np.nan

    fig, ax = plt.subplots(figsize=(8, 7))

    mesh = ax.pcolormesh(
        x,
        y,
        flux_slice.T,
        shading="auto",
        cmap="viridis"
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Neutron flux, neutrons/(cm²·s)"
    )

    ax.set_title("Neutron flux distribution (XY plane)")
    ax.set_xlabel("X, cm")
    ax.set_ylabel("Y, cm")
    ax.set_aspect("equal")

    return fig


def plot_capture_slice(field: dict, radius: float):
    """
    Plot volumetric radiative capture rate in the XY plane (z = 0).
    """

    resolution = len(field["coordinates"])
    center = resolution // 2

    x = field["coordinates"]
    y = field["coordinates"]

    capture_slice = field["capture_density"][:, :, center].copy()
    mask_slice = field["mask"][:, :, center]

    capture_slice[~mask_slice] = np.nan

    fig, ax = plt.subplots(figsize=(8, 7))

    mesh = ax.pcolormesh(
        x,
        y,
        capture_slice.T,
        shading="auto",
        cmap="inferno"
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Capture rate, neutrons/(cm³·s)"
    )

    ax.set_title("Radiative capture distribution (XY plane)")
    ax.set_xlabel("X, cm")
    ax.set_ylabel("Y, cm")
    ax.set_aspect("equal")

    return fig


def plot_relative_flux_slice(field: dict):
    """
    Plot relative neutron flux in the central XY plane.
    """

    resolution = len(field["coordinates"])
    center = resolution // 2

    x = field["coordinates"]
    y = field["coordinates"]

    flux_slice = field["flux"][:, :, center].copy()
    mask_slice = field["mask"][:, :, center]

    center_flux = field["flux"][center, center, center]

    if center_flux <= 0:
        raise ValueError("Center flux must be positive")

    relative_flux = flux_slice / center_flux
    relative_flux[~mask_slice] = np.nan

    fig, ax = plt.subplots(figsize=(8, 7))

    mesh = ax.pcolormesh(
        x,
        y,
        relative_flux.T,
        shading="auto",
        cmap="viridis"
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Relative neutron flux"
    )

    ax.set_title("Relative neutron flux distribution (XY plane)")
    ax.set_xlabel("X, cm")
    ax.set_ylabel("Y, cm")
    ax.set_aspect("equal")

    return fig
