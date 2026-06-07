"""
Generate figures from evaluation results.
"""

import argparse
import json
import numpy as np
from pathlib import Path

from meta_rl.utils.visualization import (
    plot_nees_over_time,
    plot_rmse_bar,
    plot_adaptation_dynamics,
    plot_spectrogram,
    plot_consistency_rate_vs_delta,
)


def load_results(results_dir: str) -> dict:
    """Load evaluation results from JSON files."""
    results = {}
    results_path = Path(results_dir)
    for f in results_path.glob("*.json"):
        with open(f) as fp:
            results[f.stem] = json.load(fp)
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--output_dir", type=str, default="paper/figures/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)

    # Figure 3: NEES over time (main result)
    if "nees_trajectories" in results:
        nees_dict = {
            name: np.array(vals)
            for name, vals in results["nees_trajectories"].items()
        }
        plot_nees_over_time(
            nees_dict, state_dim=6,
            title="NEES Over Time (Abrupt Noise Change at t=10s)",
            save_path=str(output_dir / "nees_over_time.pdf"),
        )

    # Figure 4: RMSE bar chart
    if "rmse_by_method" in results:
        plot_rmse_bar(
            results["rmse_by_method"],
            title="Position RMSE Across Methods",
            save_path=str(output_dir / "rmse_bar.pdf"),
        )

    # Figure 5: Adaptation dynamics
    if "adaptation_dynamics" in results:
        ad = results["adaptation_dynamics"]
        plot_adaptation_dynamics(
            q_true=np.array(ad["q_true"]),
            q_adapted=np.array(ad["q_adapted"]),
            r_true=np.array(ad["r_true"]),
            r_adapted=np.array(ad["r_adapted"]),
            save_path=str(output_dir / "adaptation_dynamics.pdf"),
        )

    # Figure 9: Consistency rate vs delta
    if "consistency_vs_delta" in results:
        cvd = results["consistency_vs_delta"]
        plot_consistency_rate_vs_delta(
            delta_values=np.array(cvd["deltas"]),
            consistency_rates={
                name: np.array(vals)
                for name, vals in cvd["rates"].items()
            },
            save_path=str(output_dir / "consistency_vs_delta.pdf"),
        )

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
