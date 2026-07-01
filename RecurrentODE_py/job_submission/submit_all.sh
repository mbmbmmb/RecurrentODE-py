#!/bin/bash
# Submit run_ode.sh once per named config in configs.json.
# Usage: bash submit_all.sh          # submit every key under .configs
#        bash submit_all.sh cox ltm  # submit a subset by name

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON="${CONFIG_JSON:-$SCRIPT_DIR/configs.json}"

command -v jq >/dev/null 2>&1 || { echo "jq is required to parse configs.json" >&2; exit 1; }

configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
    mapfile -t configs < <(jq -r '.configs | keys[]' "$CONFIG_JSON")
fi

for cfg in "${configs[@]}"; do
    echo "[submit_all] sbatch --export=CONFIG=$cfg run_ode.sh"
    sbatch --export="CONFIG=$cfg" "$SCRIPT_DIR/run_ode.sh"
done
