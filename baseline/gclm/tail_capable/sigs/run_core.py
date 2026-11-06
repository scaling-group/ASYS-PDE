"""One cold SIGS trajectory. Never submits jobs."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT.parents[1]/"sigs_upstream")
    args = parser.parse_args()
    policy = json.loads((ROOT/"policy.json").read_text())
    spec = policy["proposed_search"]
    if args.slots not in spec["slots"] or args.seed not in spec["seeds"]:
        raise ValueError("Case outside declared matrix")
    start = float(os.environ.get("ASYS_SIGS_START_MONOTONIC", time.monotonic()))
    if not 0 < start <= time.monotonic():
        raise ValueError("Invalid shell-entry clock")
    args.output.mkdir(parents=True, exist_ok=True)
    if (not args.output.is_dir()
            or any(p.name != "resources.txt" for p in args.output.iterdir())):
        # GNU time may open its resource file before starting this child.
        raise ValueError("New empty job output directory required; no resume")
    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream/"upstream"/"src"))
    import torch
    from search_core import Engine, run_case, write
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    context = dict(slots=args.slots, seed=args.seed, shell_entry_monotonic=start,
        policy=policy,
        model_service_calls=0, gpus=0, reference_access=False,
        initial_fitted_state_import=False, cold=True)
    write(args.output/"context.json", context)
    engine = Engine(upstream, policy)
    write(args.output/"library.json", dict(accepted=engine.atoms, exclusions=engine.exclusions,
                                          source="unchanged released SPATIAL_1D corpus"))
    run_case(args.output/"case", engine, args.slots, args.seed, started=start)


if __name__ == "__main__":
    main()
