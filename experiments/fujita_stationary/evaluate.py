"""Evaluate the original stationary model and its frozen coefficient state."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from model import Candidate

ROOT = Path(__file__).resolve().parent


def load():
    model = Candidate()
    model.load_state_dict(
        torch.load(ROOT / "state_dict.pt", map_location="cpu", weights_only=True), strict=True
    )
    return model.eval()


def evaluate(reference=None):
    model = load()
    a = float(model.log_origin_excess.detach())
    amplitude = math.sqrt(0.5 + math.exp(a))
    h = 0.5 - math.exp(a) / 7.5
    tail = (
        amplitude
        / math.sqrt(0.75)
        * math.exp(h + float(model.chebyshev_coefficients.detach().sum()))
    )
    eta = torch.tensor(np.unique(np.r_[np.linspace(0, 10, 10001), np.geomspace(10, 350, 1001)]))[
        :, None
    ].requires_grad_()
    value = model(eta)
    first = torch.autograd.grad(value.sum(), eta, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), eta)[0]
    # Radial regularity gives Phi'' + 4 Phi'/eta -> 5 Phi'' at the origin.
    radial = torch.where(eta == 0, 5 * second, second + 4 * first / eta.clamp_min(1e-30))
    ode = radial + value**3 - 0.5 * value - 0.5 * eta * first
    robin = eta[-1] * first[-1] + value[-1]
    result = dict(
        origin_value=amplitude,
        origin_second_derivative=-amplitude * math.exp(a) / 5,
        leading_tail_coefficient=tail,
        peak_sampled_ode_defect=float(ode[1:].detach().abs().max()),
        origin_compatibility_absolute=float(ode[0].detach().abs()),
        origin_derivative_absolute=float(first[0].detach().abs()),
        minimum_phi=float(value.detach().min()),
        endpoint_robin_defect=float(robin.detach()),
        coefficients="original selected state_dict.pt",
        derivatives="independent automatic differentiation of forward",
        diagnostic_grid=dict(
            points=eta.numel(), core=[0, 10, 10001], tail_geometric=[10, 350, 1001]
        ),
    )
    probe = torch.tensor([[0.], [.01], [.2], [1.], [10.], [100.], [350.]])
    with torch.no_grad():
        together = model(probe)
        separate = torch.cat([model(point[None]) for point in probe])
        model.train()
        training = model(probe)
        model.eval()
    result.update(batch_max_absolute_difference=float((together-separate).abs().max()),
                  mode_max_absolute_difference=float((together-training).abs().max()))
    if reference is not None:
        with np.load(reference, allow_pickle=False) as data:
            x, truth = data["eta"], data["phi"]
        if truth.shape != x.shape or not np.all(np.diff(x) > 0) or x[0] != 0 or x[-1] != 350:
            raise ValueError("Invalid stationary reference grid")
        with torch.no_grad():
            values = model(torch.tensor(x)[:, None])[:, 0].numpy()
        if not np.isfinite(values).all() or not np.isfinite(truth).all():
            raise ValueError("Non-finite profile")
        result.update(
            relative_l2_percent=100
            * float(np.sqrt(np.trapezoid((values - truth) ** 2, x) / np.trapezoid(truth**2, x))),
            max_absolute_error=float(np.max(abs(values - truth))),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluate(args.reference), indent=2, allow_nan=False) + "\n")
