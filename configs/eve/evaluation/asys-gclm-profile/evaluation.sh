#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${EVE_EVAL_LOG_ROOT:?}"
printf '%s\n' '{"score":-1000000000.0,"physics":-1000000000.0,"normalization":-1000000000.0,"refinement":-1000000000.0,"summary":"evaluation interrupted"}' > "$EVE_EVAL_LOG_ROOT/score.yaml"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ASYS_SEED="${ASYS_SEED:-0}"
# Complete process includes imports, construction, fitting, normalization,
# export, fresh reconstruction, replay and independent public assessment.
/usr/bin/time -v -o "$EVE_EVAL_LOG_ROOT/process-cost.txt" timeout --signal=KILL 1500s \
  "${ASYS_PYTHON:-python3}" "${EVE_SOLVER_ROOT:?}/asys_eval.py" --log "$EVE_EVAL_LOG_ROOT"
