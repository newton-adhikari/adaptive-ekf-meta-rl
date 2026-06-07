#!/bin/bash
# Train CC-MetaEKF agent
set -euo pipefail

CONFIG="${1:-configs/train_config.yaml}"
SEED="${2:-42}"

echo "=== CC-MetaEKF Training ==="
echo "Config: $CONFIG"
echo "Seed: $SEED"

python -m meta_rl.train \
    --config "$CONFIG" \
    --seed "$SEED"
