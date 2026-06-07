#!/bin/bash
# =============================================================
# CC-MetaEKF: One-shot script for all experiments
#
# Usage:
#   chmod +x scripts/run_all.sh
#   screen -S paper
#   ./scripts/run_all.sh 2>&1 | tee paper_run.log
# =============================================================

set -e

echo "============================================================="
echo "CC-MetaEKF Paper Experiments"
echo "Started: $(date)"
echo "============================================================="

# Check GPU
python3 -c "import torch; assert torch.cuda.is_available(), 'NO GPU!'; print(f'GPU: {torch.cuda.get_device_name()}')"

# =============================================================
# PHASE 1: Full method (ST-SIE + PID-CCPO) across 5 seeds
# This gives us Table 2 (cross-seed stability)
# =============================================================
echo ""
echo "============================================================="
echo "PHASE 1: Full method across 5 seeds"
echo "============================================================="

for seed in 42 123 456 789 1024; do
    echo ""
    echo ">>> Seed $seed starting at $(date)"
    python3 run_all.py --seed $seed --phase 1 --epochs 2000
    echo ">>> Seed $seed done at $(date)"
done

# =============================================================
# PHASE 2: Ablation matrix (seed 42 only)
# This gives us Table 4 (component contribution)
# =============================================================
echo ""
echo "============================================================="
echo "PHASE 2: Ablation (seed 42)"
echo "============================================================="

echo ">>> Ablation starting at $(date)"
python3 run_all.py --seed 42 --phase ablation --epochs 2000
echo ">>> Ablation done at $(date)"

# =============================================================
# PHASE 3: Baseline comparison (seed 42)
# This gives us Table 1 (vs classical methods)
# =============================================================
echo ""
echo "============================================================="
echo "PHASE 3: Baseline comparison (seed 42)"
echo "============================================================="

echo ">>> Comparison starting at $(date)"
python3 run_all.py --seed 42 --phase comparison
echo ">>> Comparison done at $(date)"

# =============================================================
# PHASE 4: Generate figures
# =============================================================
echo ""
echo "============================================================="
echo "PHASE 4: Generating figures"
echo "============================================================="

python3 scripts/generate_figures.py

# =============================================================
# SUMMARY
# =============================================================
echo ""
echo "============================================================="
echo "ALL DONE at $(date)"
echo "============================================================="
echo ""
echo "Results:"
echo "  Phase 1 (5 seeds): results/run_s{42,123,456,789,1024}/phase1_full.json"
echo "  Ablation:          results/run_s42/ablation_*.json"
echo "  Comparison:        results/run_s42/comparison.json"
echo "  Figures:           paper/figures/"
echo ""
 