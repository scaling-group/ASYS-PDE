"""Read the original Graveleau construction without fitting or PDE evolution."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from diagnostics import END, measure
from profile import fitted_coefficients, tail

ROOT = Path(__file__).resolve().parent


def load(state=None):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("graveleau_model", ROOT / "selected/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Candidate({})
    model.load_state_dict(torch.load(state or ROOT / "selected/state_dict.pt",
                                    map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def evaluate(state=None, quadratures=()):
    model = load(state)
    result = {"candidate_id": "solver_cff4079e5ca7", "state": str(state or ROOT / "selected/state_dict.pt"),
              "field_readout": measure(model), "normalized_front_component": []}
    for fraction in (0., .5, 1.):
        beta, coefficients = fitted_coefficients(model, fraction*END)
        result["normalized_front_component"].append(
            dict(time_fraction=fraction, beta=beta, polynomial_coefficients=coefficients,
                 **tail(beta, coefficients)))
    if quadratures:
        from contraction import integral
        result["contraction_quadratures"] = [integral(model, n) for n in quadratures]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, help="Read another state of this same construction")
    parser.add_argument("--quadrature", type=int, nargs="+", choices=(32, 64, 128), default=[],
                        help="Also compute the original L1-stability quadratures")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(evaluate(args.state, args.quadrature), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
