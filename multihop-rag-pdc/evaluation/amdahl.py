import pandas as pd
from scipy.optimize import minimize_scalar


def amdahl_theoretical(N: int, S: float) -> float:
    """Amdahl's Law: S = sequential fraction (0–1), N = number of processors."""
    return 1.0 / (S + (1.0 - S) / N)


def compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: n_processes, corpus_size, total_s
    Adds speedup_empirical column: T(N=1) / T(N) per corpus_size group.
    """
    results = []
    for corpus_size in df["corpus_size"].unique():
        subset = df[df["corpus_size"] == corpus_size].copy()
        baseline = subset[subset["n_processes"] == 1]["total_s"].values
        if len(baseline) == 0:
            continue
        t_seq = baseline[0]
        subset["speedup_empirical"] = t_seq / subset["total_s"]
        results.append(subset)
    return pd.concat(results).reset_index(drop=True)


def fit_sequential_fraction(df_speedup: pd.DataFrame) -> float:
    """
    Fit Amdahl's S by minimizing MSE across all empirical (N, speedup) pairs.
    Returns estimated sequential fraction S ∈ (0, 1).
    """
    pairs = list(zip(df_speedup["n_processes"], df_speedup["speedup_empirical"]))

    def mse(S):
        return sum((amdahl_theoretical(N, S) - sp) ** 2 for N, sp in pairs)

    result = minimize_scalar(mse, bounds=(0.01, 0.99), method="bounded")
    return float(result.x)
