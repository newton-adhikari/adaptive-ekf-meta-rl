"""
Generate figures for CC-MetaEKF.

Reads results from results/ directory and produces PDF figures.

Usage:
    python3 scripts/generate_figures.py
"""

import json, argparse, glob
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                          "font.family": "serif", "text.usetex": False})
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed — skipping figure generation")
    print("Install with: pip install matplotlib")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig1_main_results_table(results_dir, out_dir):
    """Figure: Main results bar chart (NEES + Consistency)."""
    comp_file = results_dir / "comparison.json"
    if not comp_file.exists():
        print("  Skipping fig1: no comparison.json"); return
    data = load_json(comp_file)
    if "results" in data:
        data = data["results"]

    methods = ["Fixed EKF", "Sage-Husa", "Innovation-Based", "RLS Covariance",
               "VB-EKF", "CC-MetaEKF", "Oracle EKF"]
    methods = [m for m in methods if m in data]
    colors = {"Fixed EKF": "#999", "Sage-Husa": "#c44", "Innovation-Based": "#c84",
              "RLS Covariance": "#cc4", "VB-EKF": "#4c4", "CC-MetaEKF": "#24c",
              "Oracle EKF": "#aaa"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Consistency bar chart
    cons = [data[m].get("cons", 0) * 100 for m in methods]
    bars = ax1.bar(range(len(methods)), cons,
                   color=[colors.get(m, "#888") for m in methods],
                   edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Consistency Rate (%)")
    ax1.set_title("(a) NEES Consistency (95% χ² bounds)")
    ax1.axhline(90, color="red", linestyle="--", alpha=0.5, label="Target (90%)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 100)
    for i, v in enumerate(cons):
        ax1.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=7)

    # NEES bar chart
    nees = [min(data[m].get("nees", 100), 200) for m in methods]
    ax2.bar(range(len(methods)), nees,
            color=[colors.get(m, "#888") for m in methods],
            edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Mean NEES")
    ax2.set_title("(b) Mean NEES (target = 6)")
    ax2.axhline(6, color="green", linestyle="--", alpha=0.5, label="E[NEES] = n")
    ax2.legend(fontsize=8)
    for i, v in enumerate(nees):
        ax2.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_main_results.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig1_main_results.png", bbox_inches="tight")
    print(f"  Saved fig1_main_results.pdf")
    plt.close(fig)


def fig2_nees_over_time(results_dir, out_dir):
    """Figure: NEES trajectories over time for key methods."""
    comp_file = results_dir / "comparison.json"
    if not comp_file.exists():
        print("  Skipping fig2: no comparison.json"); return
    data = load_json(comp_file)
    traces = data.get("nees_traces", {})
    if not traces:
        print("  Skipping fig2: no NEES traces"); return

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"Fixed EKF": "#999", "Sage-Husa": "#c44", "CC-MetaEKF": "#24c", "Oracle EKF": "#aaa"}
    for name in ["Fixed EKF", "Sage-Husa", "CC-MetaEKF", "Oracle EKF"]:
        if name in traces:
            t = np.array(traces[name])
            ax.plot(np.arange(len(t)) * 0.1, np.minimum(t, 100), label=name,
                    color=colors.get(name, "#888"), linewidth=1.5 if name == "CC-MetaEKF" else 1)

    ax.axhline(1.237, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(14.449, color="gray", linestyle="--", alpha=0.4, linewidth=0.8, label="χ² bounds (95%)")
    ax.axhline(6, color="green", linestyle=":", alpha=0.3, linewidth=0.8, label="E[NEES]=6")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("NEES")
    ax.set_title("NEES Over Time (averaged across test tasks)")
    ax.legend(fontsize=8); ax.set_ylim(0, 80)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_nees_over_time.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_nees_over_time.png", bbox_inches="tight")
    print(f"  Saved fig2_nees_over_time.pdf")
    plt.close(fig)


def fig3_training_curve(results_dir, out_dir):
    """Figure: Training curve (consistency over epochs)."""
    # Look for phase1 results with history
    p1_file = results_dir / "phase1_full.json"
    if not p1_file.exists():
        print("  Skipping fig3: no phase1_full.json"); return
    data = load_json(p1_file)
    history = data.get("history", [])
    if not history:
        print("  Skipping fig3: no training history"); return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    eps = [h["ep"] for h in history]
    train_cons = [h["train_cons"] * 100 for h in history]
    test_cons = [h["test_cons"] * 100 for h in history]
    train_nees = [h["train_nees"] for h in history]
    test_nees = [h["test_nees"] for h in history]

    ax1.plot(eps, train_cons, "b-", label="Train", alpha=0.7)
    ax1.plot(eps, test_cons, "r-", label="Test", alpha=0.7)
    ax1.axhline(data.get("baseline", {}).get("cons", 0.44) * 100, color="gray",
                linestyle="--", alpha=0.5, label="Baseline")
    ax1.set_ylabel("Consistency (%)"); ax1.legend(fontsize=8)
    ax1.set_title("Training Progress: CC-MetaEKF")

    ax2.plot(eps, train_nees, "b-", label="Train NEES", alpha=0.7)
    ax2.plot(eps, test_nees, "r-", label="Test NEES", alpha=0.7)
    ax2.axhline(6, color="green", linestyle=":", alpha=0.3, label="Target NEES=6")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean NEES"); ax2.legend(fontsize=8)
    ax2.set_ylim(0, min(max(test_nees) * 1.2, 200))

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_training_curve.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_training_curve.png", bbox_inches="tight")
    print(f"  Saved fig3_training_curve.pdf")
    plt.close(fig)


def fig4_ablation_matrix(results_dir, out_dir):
    """Figure: Ablation matrix heatmap."""
    variants = {}
    for f in results_dir.glob("ablation_*.json"):
        data = load_json(f)
        key = f.stem.replace("ablation_", "")
        variants[key] = data

    if len(variants) < 2:
        print(f"  Skipping fig4: only {len(variants)} ablation results"); return

    fig, ax = plt.subplots(figsize=(5, 3.5))

    matrix = np.zeros((2, 2))
    labels = np.empty((2, 2), dtype=object)
    for i, enc in enumerate(["mlp", "stsie"]):
        for j, cons in enumerate(["none", "pid"]):
            key = f"{enc}_{cons}"
            if key in variants:
                c = variants[key].get("best_cons", 0) * 100
            else:
                c = 0
            matrix[i, j] = c
            labels[i, j] = f"{c:.1f}%"

    im = ax.imshow(matrix, cmap="YlGn", vmin=30, vmax=80, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Constraint", "PID-CCPO"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["MLP Encoder", "ST-SIE Encoder"])
    ax.set_title("Ablation Matrix: Test Consistency (%)")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Consistency (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_ablation_matrix.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig4_ablation_matrix.png", bbox_inches="tight")
    print(f"  Saved fig4_ablation_matrix.pdf")
    plt.close(fig)


def fig5_multi_seed(results_base_dir, out_dir):
    """Figure: Multi-seed consistency comparison with error bars."""
    seed_results = {}
    for d in Path(results_base_dir).parent.glob("run_s*"):
        comp = d / "comparison.json"
        if comp.exists():
            data = load_json(comp)
            if "results" in data: data = data["results"]
            for method, vals in data.items():
                if method not in seed_results:
                    seed_results[method] = []
                seed_results[method].append(vals.get("cons", 0) * 100)

    if len(seed_results) < 2:
        print("  Skipping fig5: need multiple seeds"); return

    fig, ax = plt.subplots(figsize=(8, 4))
    methods = sorted(seed_results.keys())
    means = [np.mean(seed_results[m]) for m in methods]
    stds = [np.std(seed_results[m]) for m in methods]

    colors = ["#24c" if "MetaEKF" in m else "#aaa" if "Oracle" in m else "#888" for m in methods]
    ax.bar(range(len(methods)), means, yerr=stds, capsize=4,
           color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Consistency Rate (%)")
    ax.set_title(f"Consistency Across {max(len(v) for v in seed_results.values())} Seeds (mean ± std)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_multi_seed.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig5_multi_seed.png", bbox_inches="tight")
    print(f"  Saved fig5_multi_seed.pdf")
    plt.close(fig)


def main():
    if not HAS_MPL:
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/run_s42")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures from {results_dir} → {out_dir}")

    fig1_main_results_table(results_dir, out_dir)
    fig2_nees_over_time(results_dir, out_dir)
    fig3_training_curve(results_dir, out_dir)
    fig4_ablation_matrix(results_dir, out_dir)
    fig5_multi_seed(results_dir, out_dir)

    # Also generate LaTeX table
    comp_file = results_dir / "comparison.json"
    if comp_file.exists():
        data = load_json(comp_file)
        if "results" in data: data = data["results"]
        print(f"\n{'='*60}")
        print(f"{'='*60}")
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\caption{Comparison of EKF noise adaptation methods on 6D state estimation with non-stationary noise.}")
        print(r"\label{tab:main_results}")
        print(r"\begin{tabular}{lccc}")
        print(r"\toprule")
        print(r"Method & NEES $\downarrow$ & Consistency (\%) $\uparrow$ & RMSE $\downarrow$ \\")
        print(r"\midrule")
        order = ["Fixed EKF", "Sage-Husa", "Innovation-Based", "RLS Covariance",
                 "VB-EKF", "CC-MetaEKF", "Oracle EKF"]
        for m in order:
            if m not in data: continue
            d = data[m]
            bold = r"\textbf" if m == "CC-MetaEKF" else ""
            name = r"\textbf{CC-MetaEKF (ours)}" if m == "CC-MetaEKF" else m
            if m == "Oracle EKF":
                print(r"\midrule")
            nees = d.get("nees", 0)
            cons = d.get("cons", 0) * 100
            rmse = d.get("rmse", 0)
            if m == "CC-MetaEKF":
                print(f"{name} & \\textbf{{{nees:.1f}}} & \\textbf{{{cons:.1f}}} & \\textbf{{{rmse:.3f}}} \\\\")
            else:
                print(f"{name} & {nees:.1f} & {cons:.1f} & {rmse:.3f} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    print(f"\nDone. Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
