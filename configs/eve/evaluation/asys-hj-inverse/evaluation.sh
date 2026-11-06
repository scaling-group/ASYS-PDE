#!/usr/bin/env bash
set -euo pipefail
: "${EVE_SOLVER_ROOT:?Native solver workspace required}"
: "${EVE_EVAL_LOG_ROOT:?Native evaluation log directory required}"
: "${ASYS_PYTHON:?Supply the prepared scientific Python interpreter}"
mkdir -p "$EVE_EVAL_LOG_ROOT"
cd "$EVE_SOLVER_ROOT"
exec "$ASYS_PYTHON" -m problems.inverse.hj_coefficients.evaluate
