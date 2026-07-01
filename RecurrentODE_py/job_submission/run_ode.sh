#!/bin/bash
# General SLURM job submission script for every RecurrentODE_py module.
#
# Usage:
#   sbatch --export=CONFIG=<name> run_ode.sh
#
# <name> is a key under "configs" in configs.json (cox, aft, ltm, npmle,
# re_cox, re_aft, re_ltm, ...). Edit configs.json to change N, knots,
# data setting, etc. — no command-line arguments needed.
#
# Array task ID chunks the seeds: seeds = seed_offset + [(task-1)*S+1 ..
# task*S] where S = seeds_per_task (default 50). With --array=1-20 and
# seeds_per_task=50 the submission covers seeds 1..1000.

#"#SBATCH" directives that convey submission options:

#SBATCH --job-name=ode_run
#SBATCH --mail-user=bomeng@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --array=1-20
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=05-10:00:00

#SBATCH --account=gongjun0
#SBATCH --partition=standard
#SBATCH --output=/home/%u/random_effects/log/%x-%A_%a.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CONFIG_JSON="${CONFIG_JSON:-$SCRIPT_DIR/configs.json}"

: "${CONFIG:?CONFIG must be set, e.g. sbatch --export=CONFIG=cox run_ode.sh}"

command -v jq >/dev/null 2>&1 || { echo "jq is required to parse configs.json" >&2; exit 1; }

# Resolve each field: entry value wins, else default, else hard-coded fallback.
read_field() {
    local field="$1"
    local fallback="$2"
    jq -r --arg cfg "$CONFIG" --arg f "$field" --arg fb "$fallback" '
        (.configs[$cfg] // error("unknown CONFIG \($cfg)")) as $c
      | (.defaults[$f] // null)                                   as $d
      | ($c[$f] // $d // $fb)
      | if type == "array" then join(" ") else tostring end
    ' "$CONFIG_JSON"
}

MODULE=$(read_field module "")
DATA_SETTING=$(read_field data_setting "")
KNOTS=$(read_field knots "")
N_VALUES=$(read_field n_values "1000 2000")
CI=$(read_field ci "1")
SEEDS_PER_TASK=$(read_field seeds_per_task "50")
SEED_OFFSET=$(read_field seed_offset "0")

[ -n "$MODULE" ]       || { echo "config '$CONFIG' missing 'module'" >&2; exit 1; }
[ -n "$DATA_SETTING" ] || { echo "config '$CONFIG' missing 'data_setting'" >&2; exit 1; }

module load python/3.10-anaconda
cd "$REPO_ROOT"

task_id="${SLURM_ARRAY_TASK_ID:-1}"
seed_start=$(( SEED_OFFSET + (task_id - 1) * SEEDS_PER_TASK + 1 ))
seed_end=$(( SEED_OFFSET + task_id * SEEDS_PER_TASK ))

echo "[run_ode] CONFIG=$CONFIG module=$MODULE setting=$DATA_SETTING knots='${KNOTS}' ci=$CI"
echo "[run_ode] array task $task_id -> seeds $seed_start..$seed_end, N in {$N_VALUES}"

for seed in $(seq "$seed_start" "$seed_end"); do
    for N in $N_VALUES; do
        if [ -z "${KNOTS}" ]; then
            python3 -m "RecurrentODE_py.${MODULE}.main" "$N" "$seed" "$DATA_SETTING" "$CI"
        else
            python3 -m "RecurrentODE_py.${MODULE}.main" "$N" "$seed" "$DATA_SETTING" "$KNOTS" "$CI"
        fi
    done
done
