"""
CC-MetaEKF evaluation script.
This runs all baselines and ablations for different test scenarios.

python -m meta_rl.evaluate --checkpoint checkpoints/best_model.pt --config configs/eval_config.yaml
"""

import argparse
import json
import yaml
import numpy as np


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
