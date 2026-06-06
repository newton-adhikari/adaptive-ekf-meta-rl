"""
CC-MetaEKF training script.


python -m meta_rl.train --config configs/train_config.yaml --seed 42
"""

import argparse
import os
import time
import yaml
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

if __name__ == "__main__":
    main()
