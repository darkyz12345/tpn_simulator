import numpy as np


def generate_free_paths(N: int, sigma: float, seed: int | None = None) -> np.ndarray:
    if N <= 0 or sigma <= 0:
        raise ValueError("N and sigma must be positive values.")
    rng = np.random.default_rng(seed)
    xi = rng.uniform(0, 1, size=N)
    return -np.log(xi) / sigma

def analytical_probability(sigma: float, distance: float) -> float:
    if sigma <= 0 or distance < 0:
        raise ValueError("invalid parameters: sigma must be positive and distance must be non-negative.")
    return float(np.exp(-sigma * distance))

def calculate_probability(free_paths: np.ndarray, distance: float) -> dict:
    if distance < 0 or free_paths.size == 0:
        raise ValueError("Distance must be non-negative and free_paths must not be empty.")
    N = free_paths.size
    successes = np.count_nonzero(free_paths > distance)
    probability = successes / N
    standard_error = np.sqrt(probability * (1 - probability) / N)
    return {
        "N": N,
        "successes": int(successes),
        "robability": float(probability),
        "standard_error": float(standard_error)
    }

def simulate_free_path(N: int, sigma: float, distance: float, seed: int | None = None) -> dict:
    free_paths = generate_free_paths(N, sigma, seed)
    result = calculate_probability(free_paths, distance)
    result['analytical_probability'] = analytical_probability(sigma, distance)
    result["absolute_error"] = abs(result["robability"] - result["analytical_probability"])
    result["free_paths"] = free_paths
    return result