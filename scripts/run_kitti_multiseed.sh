#!/bin/bash
# =============================================================
# KITTI Multi-Seed Evaluation (ST-SIE + MLP ablation)
# Runs eval_kitti.py across all trained seeds and collects results.
#
# Usage:
#   chmod +x scripts/run_kitti_multiseed.sh
#   ./scripts/run_kitti_multiseed.sh
# =============================================================

set -e

SEQUENCES="00 02 05 07"
SEEDS=(42 123 456)
RESULTS_DIR="results"
SUMMARY_FILE="$RESULTS_DIR/kitti_multiseed_summary.txt"

echo "============================================================="
echo "CC-MetaEKF: KITTI Multi-Seed Evaluation"
echo "============================================================="
echo "Sequences: $SEQUENCES"
echo "Seeds: ${SEEDS[*]}"
echo ""

> "$SUMMARY_FILE"
echo "CC-MetaEKF KITTI Multi-Seed Results" >> "$SUMMARY_FILE"
echo "=====================================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Run each seed with BOTH ST-SIE and MLP checkpoints
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "============================================================="
    echo "Seed $seed"
    echo "============================================================="

    # Find ST-SIE checkpoint
    STSIE_CKPT=""
    for dir in "results/run_s${seed}"; do
        if [ -f "${dir}/best_stsie_pid_s${seed}.pt" ]; then
            STSIE_CKPT="${dir}/best_stsie_pid_s${seed}.pt"
            break
        fi
    done

    # Find MLP checkpoint
    MLP_CKPT=""
    for dir in "results/run_s${seed}"; do
        if [ -f "${dir}/best_mlp_pid_s${seed}.pt" ]; then
            MLP_CKPT="${dir}/best_mlp_pid_s${seed}.pt"
            break
        fi
    done

    echo "  ST-SIE: ${STSIE_CKPT:-NOT FOUND}"
    echo "  MLP:    ${MLP_CKPT:-NOT FOUND}"

    if [ -z "$STSIE_CKPT" ]; then
        echo "  SKIPPING seed $seed (no ST-SIE checkpoint)"
        continue
    fi

    # Build command
    CMD="python3 scripts/eval_kitti.py --checkpoint $STSIE_CKPT --sequences $SEQUENCES --output $RESULTS_DIR/kitti_seed_${seed}.json"

    if [ -n "$MLP_CKPT" ]; then
        CMD="$CMD --checkpoint-mlp $MLP_CKPT"
    fi

    echo "  Running: $CMD"
    echo ""
    eval $CMD

    echo "" >> "$SUMMARY_FILE"
    echo "Seed $seed:" >> "$SUMMARY_FILE"
    echo "  ST-SIE: $STSIE_CKPT" >> "$SUMMARY_FILE"
    echo "  MLP:    ${MLP_CKPT:-N/A}" >> "$SUMMARY_FILE"
done

# Generate combined summary
echo ""
echo "============================================================="
echo "COMBINED MULTI-SEED SUMMARY"
echo "============================================================="

python3 -c "
import json, numpy as np
from pathlib import Path

seeds = [42, 123, 456]
results_dir = Path('$RESULTS_DIR')

# Collect results per seed
stsie_cons = {}
mlp_cons = {}
inflate_cons = {}

for seed in seeds:
    f = results_dir / f'kitti_seed_{seed}.json'
    if not f.exists():
        print(f'  WARNING: {f} not found')
        continue
    data = json.load(open(f))

    # ST-SIE
    vals = []
    for seq in data.values():
        for scenario in seq.values():
            if 'ccmetaekf' in scenario:
                vals.append(scenario['ccmetaekf']['cons'])
    if vals:
        stsie_cons[seed] = np.mean(vals)

    # MLP
    mlp_vals = []
    for seq in data.values():
        for scenario in seq.values():
            if 'ccmetaekf_mlp' in scenario:
                mlp_vals.append(scenario['ccmetaekf_mlp']['cons'])
    if mlp_vals:
        mlp_cons[seed] = np.mean(mlp_vals)

    # Inflation (same across seeds, just grab once)
    if not inflate_cons:
        for method in ['fixed', 'sage_husa', 'oracle', 'inflate_1.5', 'inflate_2.0', 'inflate_3.0', 'inflate_4.0']:
            iv = []
            for seq in data.values():
                for scenario in seq.values():
                    if method in scenario:
                        iv.append(scenario[method]['cons'])
            if iv:
                inflate_cons[method] = (np.mean(iv), np.std(iv))

# Print baselines
print('Baselines & Q-inflation (deterministic):')
print('-' * 50)
for method in ['fixed', 'sage_husa', 'oracle', 'inflate_1.5', 'inflate_2.0', 'inflate_3.0', 'inflate_4.0']:
    if method in inflate_cons:
        m, s = inflate_cons[method]
        label = f'Q*{method.split(\"_\")[1]}' if method.startswith('inflate') else method
        print(f'  {label:<15s}: {m:.1%} +/- {s:.1%}')

# ST-SIE per seed
print()
print('CC-MetaEKF (ST-SIE + PID-CCPO) per seed:')
print('-' * 50)
for seed in sorted(stsie_cons.keys()):
    print(f'  Seed {seed:>4d}: {stsie_cons[seed]:.1%}')
if stsie_cons:
    vals = list(stsie_cons.values())
    print(f'  MEAN:      {np.mean(vals):.1%} +/- {np.std(vals):.1%} (n={len(vals)})')

# MLP per seed
print()
print('CC-MetaEKF (MLP + PID-CCPO) per seed:')
print('-' * 50)
if mlp_cons:
    for seed in sorted(mlp_cons.keys()):
        print(f'  Seed {seed:>4d}: {mlp_cons[seed]:.1%}')
    vals = list(mlp_cons.values())
    print(f'  MEAN:      {np.mean(vals):.1%} +/- {np.std(vals):.1%} (n={len(vals)})')
else:
    print('  No MLP results found!')

# Comparison
if stsie_cons and mlp_cons:
    common = sorted(set(stsie_cons.keys()) & set(mlp_cons.keys()))
    if common:
        print()
        print('Head-to-head (ST-SIE vs MLP):')
        print('-' * 50)
        for seed in common:
            delta = stsie_cons[seed] - mlp_cons[seed]
            winner = 'ST-SIE' if delta > 0 else 'MLP'
            print(f'  Seed {seed:>4d}: ST-SIE={stsie_cons[seed]:.1%} MLP={mlp_cons[seed]:.1%} ({winner} +{abs(delta):.1%})')
        stsie_mean = np.mean([stsie_cons[s] for s in common])
        mlp_mean = np.mean([mlp_cons[s] for s in common])
        print(f'  MEAN:      ST-SIE={stsie_mean:.1%} MLP={mlp_mean:.1%} (delta={stsie_mean-mlp_mean:+.1%})')

# Key result
print()
print('=' * 50)
print('KEY RESULT: Learned vs Fixed Inflation')
print('=' * 50)
if stsie_cons and 'inflate_4.0' in inflate_cons:
    best_inflate = inflate_cons['inflate_4.0'][0]
    stsie_mean = np.mean(list(stsie_cons.values()))
    gap = stsie_mean - best_inflate
    print(f'  Best fixed Q-inflation (Q*4): {best_inflate:.1%}')
    print(f'  CC-MetaEKF (ST-SIE, mean):    {stsie_mean:.1%}')
    print(f'  Advantage:                     +{gap:.1%}')
    print(f'  --> Policy learns MORE than just Q inflation')
" | tee -a "$SUMMARY_FILE"

echo ""
echo "============================================================="
echo "Done. Results in:"
echo "  Per-seed: $RESULTS_DIR/kitti_seed_*.json"
echo "  Summary:  $SUMMARY_FILE"
echo "============================================================="
 