#!/usr/bin/env bash
set -euo pipefail
: "${EVE_SOLVER_ROOT:?Native solver workspace required}"
: "${ASYS_PYTHON:?Supply the prepared scientific Python interpreter}"

mkdir -p "$EVE_EVAL_LOG_ROOT"

score_path="$EVE_EVAL_LOG_ROOT/score.yaml"
error_path="$EVE_EVAL_LOG_ROOT/error.txt"
tmp_stdout="$(mktemp)"
tmp_stderr="$(mktemp)"
cleanup() {
  rm -f "$tmp_stdout" "$tmp_stderr"
}
trap cleanup EXIT

problem="${ASYS_PROBLEM:?ASYS_PROBLEM not set}"
device="${ASYS_DEVICE:-cpu}"
evaluate_path="problems/${problem}/evaluate.py"

export ASYS_DEVICE="$device"
eval_artifact_dir="$EVE_EVAL_LOG_ROOT/eval_artifacts"
mkdir -p "$eval_artifact_dir"
export ASYS_LOSS_HISTORY="$eval_artifact_dir/loss_history.npz"
export ASYS_EVAL_REPORT="$EVE_EVAL_LOG_ROOT/eval_report.txt"

# Unified training budget for all ASYS cases (rep=1): a single 300s wall-clock
# cap. This shared script is the authoritative source; it overrides any
# per-case ASYS_WALL_TIME default set in the case evaluate_*.sh.
export ASYS_WALL_TIME=300

# Wall-time early stop for all ASYS cases: stop once best_loss has not improved
# for 30% of the budget (after a 15% warmup). The evaluator-owned training loop
# enforces this on real wall time, so an analytic-solution candidate that
# converges in seconds does not burn the full budget, while a still-improving
# candidate runs to the end.
export ASYS_EARLYSTOP_PATIENCE_FRAC="${ASYS_EARLYSTOP_PATIENCE_FRAC:-0.3}"

status=0
(
  cd "$EVE_SOLVER_ROOT"

  if [[ ! -f "$evaluate_path" ]]; then
    printf 'Missing evaluator: %s\n' "$evaluate_path" >&2
    exit 2
  fi
  "$ASYS_PYTHON" "$evaluate_path" --json --save-state-dict "$eval_artifact_dir/state_dict.pt"
) >"$tmp_stdout" 2>"$tmp_stderr" || status=$?

if [[ -s "$tmp_stdout" ]]; then
  cat "$tmp_stdout"
fi
if [[ -s "$tmp_stderr" ]]; then
  cat "$tmp_stderr" >&2
fi

if [[ $status -ne 0 ]]; then
  {
    printf 'Evaluation command failed with exit code %s\n' "$status"
    if [[ -s "$tmp_stderr" ]]; then
      printf '\n'
      cat "$tmp_stderr"
    fi
  } >"$error_path"
  exit "$status"
fi

"$ASYS_PYTHON" - "$tmp_stdout" "$score_path" "$error_path" <<'PY'
import json
import sys
from pathlib import Path

stdout_path = Path(sys.argv[1])
score_path = Path(sys.argv[2])
error_path = Path(sys.argv[3])

raw = stdout_path.read_text(encoding="utf-8").strip()
try:
    payload = json.loads(raw)
    payload["score"] = float(payload["score"])
except Exception as exc:
    error_path.write_text(f"Failed to parse evaluator output: {exc}\n\nRaw stdout:\n{raw}\n", encoding="utf-8")
    raise SystemExit(1)

score_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
PY
