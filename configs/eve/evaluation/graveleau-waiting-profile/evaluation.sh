#!/usr/bin/env bash
set -euo pipefail
: "${EVE_SOLVER_ROOT:?}"
: "${EVE_EVAL_LOG_ROOT:?}"
evaluation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${EVE_EVAL_LOG_ROOT}"
printf '%s\n' '{"score":-1000000000,"physics":-1000000000,"ic":-1000000000,"ic_tangent":-1000000000,"bc":-1000000000,"summary":"Conditional profile evaluation incomplete"}' > "${EVE_EVAL_LOG_ROOT}/score.yaml"
"${ASYS_PYTHON:-python3}" "${evaluation_dir}/qualify.py" --application "${EVE_SOLVER_ROOT}" --output "${EVE_EVAL_LOG_ROOT}/qualification.json" > "${EVE_EVAL_LOG_ROOT}/qualification-stdout.log" 2> "${EVE_EVAL_LOG_ROOT}/qualification-stderr.log" || exit 0
"${ASYS_PYTHON:-python3}" "${evaluation_dir}/fit.py" \
  --solver "${EVE_SOLVER_ROOT}" --output "${EVE_EVAL_LOG_ROOT}/fit" \
  --judge corrected --seconds "${ASYS_GRAVELEAU_WALL_SECONDS:-300}" \
  > "${EVE_EVAL_LOG_ROOT}/stdout.log" 2> "${EVE_EVAL_LOG_ROOT}/stderr.log" || exit 0
"${ASYS_PYTHON:-python3}" "${evaluation_dir}/score.py" \
  --fit "${EVE_EVAL_LOG_ROOT}/fit" --output "${EVE_EVAL_LOG_ROOT}/fit/report.json" \
  --score "${EVE_EVAL_LOG_ROOT}/score.yaml" \
  > "${EVE_EVAL_LOG_ROOT}/scoring-stdout.log" 2> "${EVE_EVAL_LOG_ROOT}/scoring-stderr.log" || exit 0
