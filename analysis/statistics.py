import numpy as np


def convergence_analysis(N_values: list[int],
                         sigma: float,
                         distance: float,
                         repeats: int = 100,
                         seed: int | None = None) -> dict:
    if sigma <= 0 or distance < 0:
        raise ValueError("invalid parameters: sigma must be positive and distance must be non-negative.")
    if repeats < 2 or not N_values or any(n <= 0 for n in N_values):
        raise ValueError("repeats must be at least 2 and N_values must contain positive integers.")
    rng = np.random.default_rng(seed)
    analytical = np.exp(-sigma * distance)
    empirical_errors = []
    theoretical_errors = []
    for N in N_values:
        probabilities = []
        for _ in range(repeats):
            xi = rng.uniform(0, 1, size=N)
            free_paths = -np.log(xi) / sigma
            p = np.mean(free_paths > distance)
            probabilities.append(p)
        empirical_error = np.std(probabilities, ddof=1)
        theoretical_error = np.sqrt(analytical * (1 - analytical) / N)
        empirical_errors.append(empirical_error)
        theoretical_errors.append(theoretical_error)
    return {
        "N_values": np.asarray(N_values),
        "empirical_errors": np.asarray(empirical_errors),
        "theoretical_errors": np.asarray(theoretical_errors),
    }