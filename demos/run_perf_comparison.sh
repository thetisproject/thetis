#!/bin/bash
# Run all North Sea performance comparison configurations and collect results.
#
# Usage (from the demos/ directory):
#
#   bash run_perf_comparison.sh            # run all five
#   bash run_perf_comparison.sh 0 1 3      # run subset by index
#
# Output is written to perf_results.txt in the current directory, with
# stdout/stderr from each run also printed to the terminal.
#
# The TPXO data directory is expected at $DATA/tpxo or ./data/tpxo.
# The spinup files must be present in outputs_spinup/.

set -euo pipefail

SCRIPTS=(
    "perf_baseline.py"           # 0 baseline
    "perf_1_dg_cg.py"            # 1 dg-cg + PressureProjectionPicard
    "perf_2_rt_dg.py"            # 2 rt-dg RT1/DG0 + DIRK22
    "perf_3_dg_dg_schur.py"      # 3 dg-dg + Schur/AssembledSchurPC
    "perf_4_rt_hybridization.py" # 4 rt-dg + HybridizationPC
)

LABELS=(
    "Baseline  : dg-dg P1DG + DIRK22 (default)"
    "Option 1  : dg-cg P1DG/P2CG + PressureProjectionPicard"
    "Option 2  : rt-dg RT1/DG0 + DIRK22"
    "Option 3  : dg-dg P1DG + DIRK22 + Schur/AssembledSchurPC"
    "Option 4  : rt-dg RT1/DG0 + DIRK22 + HybridizationPC"
)

# Which indices to run (default: all)
if [ $# -gt 0 ]; then
    INDICES=("$@")
else
    INDICES=(0 1 2 3 4)
fi

RESULTS_FILE="perf_results.txt"
echo "North Sea performance comparison — $(date)" > "$RESULTS_FILE"
echo "================================================" >> "$RESULTS_FILE"

for idx in "${INDICES[@]}"; do
    script="${SCRIPTS[$idx]}"
    label="${LABELS[$idx]}"
    echo ""
    echo ">>> Running $label"
    echo "    Script: $script"
    echo ""

    start=$(date +%s%N)
    python "$script" 2>&1 | tee -a "$RESULTS_FILE"
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000000 ))

    echo "    Wall clock (shell): ${elapsed}s" | tee -a "$RESULTS_FILE"
    echo "---" >> "$RESULTS_FILE"
done

echo ""
echo "Results also written to $RESULTS_FILE"
