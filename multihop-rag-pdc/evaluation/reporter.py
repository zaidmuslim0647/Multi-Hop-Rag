import os
import matplotlib.pyplot as plt
import pandas as pd
from evaluation.amdahl import amdahl_theoretical, compute_speedup, fit_sequential_fraction
from config import PLOT_DIR


def _ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_amdahl(df: pd.DataFrame):
    """
    Plot 1: Empirical vs Amdahl theoretical speedup curves.
    """
    _ensure_plot_dir()
    df_speedup = compute_speedup(df)
    S_fitted = fit_sequential_fraction(df_speedup)

    N_range = sorted(df["n_processes"].unique())
    theoretical = [amdahl_theoretical(N, S_fitted) for N in N_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    for corpus_size, group in df_speedup.groupby("corpus_size"):
        group_sorted = group.sort_values("n_processes")
        ax.plot(
            group_sorted["n_processes"],
            group_sorted["speedup_empirical"],
            marker="o",
            label=f"Empirical ({corpus_size:,} docs)",
        )

    ax.plot(N_range, theoretical, "k--", linewidth=1.5, label=f"Amdahl (S={S_fitted:.2f})")
    ax.set_xlabel("Number of processes (N)")
    ax.set_ylabel("Speedup (T_seq / T_par)")
    ax.set_title("Empirical vs Theoretical Speedup (Amdahl's Law)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(PLOT_DIR, "amdahl_speedup.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return S_fitted


def plot_latency_vs_corpus(df: pd.DataFrame):
    """
    Plot 2: Preprocessing total wall-clock time vs corpus size (log scale).
    """
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_proc, group in df.groupby("n_processes"):
        group_sorted = group.sort_values("corpus_size")
        ax.plot(
            group_sorted["corpus_size"],
            group_sorted["total_s"],
            marker="o",
            label=f"N={n_proc}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Corpus size (documents, log scale)")
    ax.set_ylabel("Total preprocessing time (seconds)")
    ax.set_title("Preprocessing Latency vs Corpus Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(PLOT_DIR, "latency_vs_corpus.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_accuracy_comparison(baseline_results: dict, multihop_results: dict):
    """
    Plot 3: Grouped bar chart comparing FinQA EM and MultiHop F1 across phases.
    baseline_results / multihop_results: {"finqa_em": float, "multihop_f1": float}
    """
    _ensure_plot_dir()
    metrics = ["FinQA EM", "MultiHop-RAG F1"]
    baseline_vals = [baseline_results.get("finqa_em", 0), baseline_results.get("multihop_f1", 0)]
    multihop_vals = [multihop_results.get("finqa_em", 0), multihop_results.get("multihop_f1", 0)]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], baseline_vals, width, label="Phase 1 — Baseline")
    ax.bar([i + width / 2 for i in x], multihop_vals, width, label="Phase 2 — Multi-Hop")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy: Baseline vs Multi-Hop RAG")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    out = os.path.join(PLOT_DIR, "accuracy_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_results_table(baseline: dict, multihop: dict, bench_df: pd.DataFrame):
    """Print the final results table for the paper."""
    bench_10k = bench_df[bench_df["corpus_size"] == 5000] if 5000 in bench_df["corpus_size"].values else bench_df
    t_seq = bench_10k[bench_10k["n_processes"] == 1]["total_s"].values
    t_par_max = bench_10k[bench_10k["n_processes"] == bench_10k["n_processes"].max()]["total_s"].values

    t_seq_str = f"{t_seq[0]:.1f}s" if len(t_seq) else "—"
    t_par_str = f"{t_par_max[0]:.1f}s" if len(t_par_max) else "—"
    speedup = f"{t_seq[0]/t_par_max[0]:.2f}×" if len(t_seq) and len(t_par_max) else "—"

    print("\n=== Final Results Table ===")
    print(f"{'Phase':<45} {'FinQA EM':>10} {'MultiHop F1':>12} {'T(N=1)':>8} {'T(N=max)':>10} {'Speedup':>8}")
    print("-" * 95)
    print(f"{'Phase 1 — Sequential Baseline':<45} {baseline.get('finqa_em', 0):>10.3f} {baseline.get('multihop_f1', 0):>12.3f} {t_seq_str:>8} {'—':>10} {'1×':>8}")
    print(f"{'Phase 2 — Multi-Hop (Sequential Infra)':<45} {multihop.get('finqa_em', 0):>10.3f} {multihop.get('multihop_f1', 0):>12.3f} {t_seq_str:>8} {'—':>10} {'1×':>8}")
    print(f"{'Phase 3 — Multi-Hop (Parallel Infra)':<45} {multihop.get('finqa_em', 0):>10.3f} {multihop.get('multihop_f1', 0):>12.3f} {'—':>8} {t_par_str:>10} {speedup:>8}")
