#!/usr/bin/env bash
set -euo pipefail
export ASYS_PROBLEM=pinn/allen_cahn_2d
export ASYS_DEVICE="${ASYS_DEVICE:-cpu}"
export ASYS_WALL_TIME="${ASYS_WALL_TIME:-120}"
exec bash "$(dirname "$0")/evaluation.sh"
