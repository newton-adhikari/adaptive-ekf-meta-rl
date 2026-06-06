#!/usr/bin/env python3
"""
CC-MetaEKF: One-command training pipeline.

runs all experiments, saves results + logs.

"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import time, argparse, json, os, sys, warnings, logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")


# ================================================================
# Setup
# ================================================================

def setup_device():
    if torch.cuda.is_available():
        dev = "cuda"
        name = torch.cuda.get_device_name()
        torch.backends.cudnn.benchmark = True
    else:
        dev = "cpu"
        name = f"{os.cpu_count()} CPU cores"
        torch.set_num_threads(os.cpu_count() or 1)
    return dev, name

DEVICE, DEVICE_NAME = setup_device()

def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"run_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger("ccmetaekf"), log_file

def log_and_save(results, name, output_dir):
    path = output_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return path