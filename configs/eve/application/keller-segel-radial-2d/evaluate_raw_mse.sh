#!/usr/bin/env bash
set -euo pipefail
: "${EVE_SOLVER_ROOT:?}"
: "${EVE_EVAL_LOG_ROOT:?}"
export ASYS_PYTHON="${ASYS_PYTHON:-python3}"
mkdir -p "${EVE_EVAL_LOG_ROOT}/eval_artifacts"
printf '%s\n' '{"score":-1000000000.0,"physics":-1000000000.0,"ic":-1000000000.0,"ic_tangent":-1000000000.0,"bc":-1000000000.0,"summary":"KS raw-mse evaluation did not complete"}' > "${EVE_EVAL_LOG_ROOT}/score.yaml"
export ASYS_WALL_TIME=300 ASYS_EARLYSTOP_PATIENCE_FRAC=0.3
export PYTHONDONTWRITEBYTECODE=1
"${ASYS_PYTHON}" "$(dirname "$0")/evaluate.py" fit raw-mse > "${EVE_EVAL_LOG_ROOT}/fitting_stdout.log" 2> "${EVE_EVAL_LOG_ROOT}/fitting_stderr.log"
"${ASYS_PYTHON}" "$(dirname "$0")/evaluate.py" score raw-mse > "${EVE_EVAL_LOG_ROOT}/stdout.log" 2> "${EVE_EVAL_LOG_ROOT}/stderr.log"
