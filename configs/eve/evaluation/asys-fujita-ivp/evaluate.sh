#!/usr/bin/env bash
set -euo pipefail
export ASYS_PROBLEM=blowup/fujita_radial_n5_p3
export ASYS_DEVICE="${ASYS_DEVICE:-cpu}"
exec bash "$(dirname "$0")/evaluation.sh"
