#!/bin/bash
# SLURM submission for the Python LTM simulation
# (RecurrentODE_py/ltm/main.py).
#
# Usage:
#   sbatch ltm.sh
#
# Array layout: --array=1-20 with SEEDS_PER_TASK=5 covers seeds 1..100
# (matching MATLAB ltm/summary.m: N=1000, rep=100, data_setting=4, K4).
# Each task runs SEEDS_PER_TASK seeds sequentially, for every N in N_VALUES.

#SBATCH --job-name=ode_ltm
#SBATCH --mail-user=bomeng@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --array=1-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=05-10:00:00

#SBATCH --account=gongjun0
#SBATCH --partition=standard
#SBATCH --output=/home/%u/random_effects/log/%x-%A_%a.log

set -euo pipefail

# ---- LTM-specific settings (match ltm/summary.m) --------------------------
MODULE="ltm"
DATA_SETTING=4
KNOTS="K4"
N_VALUES=(1000)
CI=1
SEEDS_PER_TASK=5         # 20 array tasks * 5 seeds = 100 reps
SEED_OFFSET=0
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

module load python/3.10-anaconda
cd "$REPO_ROOT"

task_id="${SLURM_ARRAY_TASK_ID:-1}"
seed_start=$(( SEED_OFFSET + (task_id - 1) * SEEDS_PER_TASK + 1 ))
seed_end=$(( SEED_OFFSET + task_id * SEEDS_PER_TASK ))

echo "[ltm.sh] task=$task_id seeds $seed_start..$seed_end  N=${N_VALUES[*]}  knots=$KNOTS  ci=$CI"

for seed in $(seq "$seed_start" "$seed_end"); do
    for N in "${N_VALUES[@]}"; do
        python3 -m "RecurrentODE_py.${MODULE}.main" \
            "$N" "$seed" "$DATA_SETTING" "$KNOTS" "$CI"
    done
done
