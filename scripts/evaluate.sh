#!/bin/bash
# Evaluate CC-MetaEKF and all baselines
set -euo pipefail

CHECKPOINT="${1:-checkpoints/best_model.pt}"
CONFIG="${2:-configs/eval_config.yaml}"

echo "=== CC-MetaEKF Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"

python -m meta_rl.evaluate \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG"
