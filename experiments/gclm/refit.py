"""Refit one returned ASYS family with the shared support-aware fitter."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from cohort_readout import ROOT, TASK, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", choices=("H1", "H2", "H3", "H4", "H5"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = next(row for row in manifest()["rows"] if row["id"] == "asys-" + args.history.lower())
    source = ROOT / row["folder"]
    config = json.loads((source / "model_config.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    # The protected evaluator expects a case-local source layout. Stage only
    # that layout; fitting code and numerical policy remain shared.
    with tempfile.TemporaryDirectory(prefix="gclm-refit-") as temporary:
        case = Path(temporary)
        shutil.copyfile(source / "model.py", case / "model.py")
        shutil.copyfile(source / "model_config.json", case / "config.json")
        for name in ("fit_policy.json", "assessment_config.json"):
            shutil.copyfile(TASK / name, case / name)
        sys.path.insert(0, str(TASK))
        spec = importlib.util.spec_from_file_location("gclm_shared_fit", TASK / "asys_eval.py")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        evaluator.ROOT = case
        os.environ["ASYS_SEED"] = str(config["seed"])
        result = evaluator.evaluate(args.output.resolve())
        evaluator.write(args.output / "score.yaml", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
