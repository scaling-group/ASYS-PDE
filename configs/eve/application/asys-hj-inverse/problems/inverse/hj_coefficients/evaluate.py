"""Native entrypoint and quick interface qualification; neither launches a search."""
import argparse
import json
from pathlib import Path

import torch
from asys_task.evaluate import load_module, validate_sources
from problems.inverse.hj_coefficients import problem as fixed


def qualify():
    case = Path(__file__).resolve().parent
    public = json.loads((case / "problem.json").read_text())
    validate_sources(case)
    field = fixed.Field(load_module(case / "model.py", "hj_check").Candidate(public), public)
    fixed.probe(field, 0.5)
    data = fixed.observations(case / "observations.npz", public["horizon"])
    print(json.dumps({"check_ok": True, "train_observations": len(data["train_points"]),
                      "validation_observations": len(data["validation_points"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        qualify()
    else:
        fixed.main()
