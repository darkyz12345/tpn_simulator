import numpy as np


BEAM_DIRECTIONS = {
    "+X": (1, 0, 0),
    "-X": (-1, 0, 0),
    "+Y": (0, 1, 0),
    "-Y": (0, -1, 0),
    "+Z": (0, 0, 1),
    "-Z": (0, 0, -1)
}

def generate_impact_points(
    radius: float,
    N: int,
    rng: np.random.Generator
) -> np.ndarray:
    if radius <= 0:
        raise ValueError("Radius must be positive")
    if N <= 0:
        raise ValueError("N must be positive")

    xi = rng.random(N)
    phi = 2 * np.pi * rng.random(N)

    r = radius * np.sqrt(xi)

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return np.column_stack((x, y))

def calculate_chord_length(impact_points: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError("Radius mus be positive")
    if impact_points.ndim != 2 or impact_points.shape[1] != 2:
        raise ValueError("Expected array with shape (N, 2)")
    if impact_points.shape[0] == 0:
        raise ValueError("impact points cannot be empty")
    r_squared = np.sum(impact_points ** 2, axis = 1)
    if np.any(r_squared > radius ** 2 *(1 + 1e-12)):
        raise ValueError("Some points lie outside the sphere projection")
    chord_length = 2 * np.sqrt(np.maximum(0, radius ** 2 - r_squared))
    return chord_length

def generate_six_beam_chords(radius: float, N_per_beam: int, seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    for beam_name in BEAM_DIRECTIONS:
        points = generate_impact_points(
            radius=radius,
            N=N_per_beam,
            rng=rng
        )
        lengths = calculate_chord_length(
            impact_points=points,
            radius=radius
            )
        results[beam_name] = {
            "impact_points": points,
            "chord_lengths": lengths
        }
    return results


def simulate_absorption(
    chord_results: dict,
    sigma: float,
    radius: float,
    intensity: float,
    seed: int | None = None
) -> dict:
    """
    Simulate neutron radiative capture in a spherical Be target.

    chord_results: output of generate_six_beam_chords()
    sigma: macroscopic capture cross-section, cm^-1
    radius: sphere radius, cm
    intensity: intensity of one beam, neutrons/(cm^2*s)
    """

    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    if radius <= 0:
        raise ValueError("Radius must be positive")
    if intensity <= 0:
        raise ValueError("Intensity must be positive")

    rng = np.random.default_rng(seed)

    results = {}
    total_neutrons = 0
    total_captured = 0
    total_expected = 0.0

    for beam_name, data in chord_results.items():
        lengths = data["chord_lengths"]

        # Generate free paths until radiative capture
        xi = rng.random(len(lengths))
        free_paths = -np.log(xi) / sigma

        # Determine whether capture occurs inside the sphere
        captured = free_paths < lengths

        N = len(lengths)
        captured_count = int(np.count_nonzero(captured))

        probability = captured_count / N

        # Analytical probability for the generated chord sample
        analytical_probability = float(
            np.mean(-np.expm1(-sigma * lengths))
        )

        standard_error = np.sqrt(
            probability * (1 - probability) / N
        )

        results[beam_name] = {
            "N": N,
            "captured": captured_count,
            "transmitted": N - captured_count,
            "probability": float(probability),
            "analytical_probability": analytical_probability,
            "standard_error": float(standard_error)
        }

        total_neutrons += N
        total_captured += captured_count
        total_expected += analytical_probability * N

    # Combined results for all six beams
    total_probability = total_captured / total_neutrons
    total_analytical_probability = total_expected / total_neutrons

    total_standard_error = np.sqrt(
        total_probability * (1 - total_probability) / total_neutrons
    )

    # Physical incident neutron rate
    incident_rate = 6 * intensity * np.pi * radius**2

    # Physical radiative capture rate
    capture_rate = incident_rate * total_probability

    return {
        "beams": results,
        "total_N": total_neutrons,
        "total_captured": total_captured,
        "total_transmitted": total_neutrons - total_captured,
        "probability": float(total_probability),
        "analytical_probability": float(total_analytical_probability),
        "standard_error": float(total_standard_error),
        "incident_rate": float(incident_rate),
        "capture_rate": float(capture_rate)
    }


def analytical_capture_probability(radius: float, sigma: float) -> float:
    """Analytical capture probability for a uniform beam."""
    if radius <= 0 or sigma <= 0:
        raise ValueError("Radius and sigma must be positive")

    a = sigma * radius

    # Series expansion avoids numerical cancellation for small a
    if a < 0.01:
        probability = (
            4 * a / 3
            - a**2
            + 8 * a**3 / 15
            - 2 * a**4 / 9
            + 8 * a**5 / 105
        )
    else:
        x = 2 * a
        transmission = (
            1 - (1 + x) * np.exp(-x)
        ) / (2 * a**2)

        probability = 1 - transmission

    return float(probability)


def analytical_capture_rate(
    radius: float,
    sigma: float,
    intensity: float
) -> float:
    """Analytical radiative capture rate, neutrons/s."""

    probability = analytical_capture_probability(radius, sigma)
    incident_rate = 6 * intensity * np.pi * radius**2

    return float(incident_rate * probability)


def calculate_flux_field(
    radius: float,
    sigma: float,
    intensity: float,
    resolution: int = 80
) -> dict:
    """
    Calculate neutron flux and volumetric radiative capture
    rate inside a spherical target.

    All distances are in cm.
    intensity: neutrons/(cm^2*s) per beam.
    sigma: macroscopic capture cross-section, cm^-1.
    """

    if radius <= 0:
        raise ValueError("Radius must be positive")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    if intensity <= 0:
        raise ValueError("Intensity must be positive")
    if resolution < 2:
        raise ValueError("Resolution must be >= 2")

    # Cartesian grid
    coordinates = np.linspace(-radius, radius, resolution)
    dx = coordinates[1] - coordinates[0]

    X, Y, Z = np.meshgrid(
        coordinates,
        coordinates,
        coordinates,
        indexing="ij"
    )

    r_squared = X**2 + Y**2 + Z**2
    mask = r_squared <= radius**2

    # Calculate flux only inside the sphere
    flux = np.zeros_like(X, dtype=float)

    sqrt_term = np.sqrt(
        np.maximum(0, radius**2 - r_squared)
    )

    # Distance from the entrance surface for each beam
    distances = {
        "+X": X + sqrt_term,
        "-X": -X + sqrt_term,
        "+Y": Y + sqrt_term,
        "-Y": -Y + sqrt_term,
        "+Z": Z + sqrt_term,
        "-Z": -Z + sqrt_term
    }

    for distance in distances.values():
        flux[mask] += intensity * np.exp(
            -sigma * distance[mask]
        )

    # Volumetric radiative capture rate
    capture_density = sigma * flux

    capture_density[~mask] = 0.0

    # Numerical volume integration
    integrated_capture_rate = float(
        np.sum(capture_density) * dx**3
    )

    sphere_volume = 4 / 3 * np.pi * radius**3

    return {
        "coordinates": coordinates,
        "X": X,
        "Y": Y,
        "Z": Z,
        "mask": mask,
        "flux": flux,
        "capture_density": capture_density,
        "dx": float(dx),
        "sphere_volume": float(sphere_volume),
        "integrated_capture_rate": integrated_capture_rate
    }
