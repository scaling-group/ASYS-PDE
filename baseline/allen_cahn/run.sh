#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1:---help}" in
    common1200) workspace=common; seconds=1200 ;;
    dense1200) workspace=dense; seconds=1200 ;;
    dense2400) workspace=dense; seconds=2400 ;;
    *) echo "Usage: $0 {common1200|dense1200|dense2400} OUTPUT_ROOT [SEED]"; exit 2 ;;
esac
variant=$1
output=${2:?Supply an output directory}
seeds=(1101101 1101102 1101103)
if [[ $# -ge 3 ]]; then seeds=("$3"); fi
export ASYS_NUM_THREADS="${ASYS_NUM_THREADS:-1}"
export ASYS_WALL_TIME="${ASYS_WALL_TIME:-$seconds}"
export PYTHONDONTWRITEBYTECODE=1
for seed in "${seeds[@]}"; do
    log="$output/$variant-seed$seed"
    if [[ -e "$log" ]]; then
        echo "Output already exists: $log" >&2
        exit 1
    fi
    "${ASYS_PYTHON:-python3}" -B "$root/$workspace/evaluate.py" \
        --fit --workspace "$root/$workspace" --log "$log" --seed "$seed"
    test ! -f "$log/error.txt"
done
