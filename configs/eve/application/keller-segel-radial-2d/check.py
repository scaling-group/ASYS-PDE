"""Small coordinate/parameter derivative guard, independent of formal fitting."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from problem import _build_model, _grad, _load_module


def check(model, *, require_parameter_sensitivity=True):
    points = torch.tensor([[0.003, 0.866], [0.075, 0.6], [0.7, 0.11], [2.3, 0.02], [12.7, 0.83]])
    coords = points.clone().requires_grad_(True)
    values = model(coords)
    assert values.shape == (len(points), 1) and torch.isfinite(values).all()
    derivative = _grad(values, coords)
    singles = torch.cat([model(row[None]) for row in points])
    batch_error = float((values - singles).detach().abs().max())
    assert torch.allclose(values, singles, atol=1e-10, rtol=1e-10), "Batch-dependent field"
    errors = []
    for axis in range(2):
        shift = torch.zeros_like(points)
        shift[:, axis] = 1e-6
        fd = (model(points + shift) - model(points - shift))[:, 0] / 2e-6
        expected = derivative[:, axis]
        errors.append(float(((fd - expected).abs() / (1 + expected.abs())).detach().max()))
        assert torch.allclose(fd, expected, atol=1e-5, rtol=1e-3), "Incorrect coordinate derivative"
    second = _grad(derivative[:, :1], coords)[:, 0]
    shift = torch.zeros_like(points)
    shift[:, 0] = 1e-5
    right = (points + shift).requires_grad_(True)
    left = (points - shift).requires_grad_(True)
    second_fd = (_grad(model(right), right)[:, 0] - _grad(model(left), left)[:, 0]) / 2e-5
    second_error = float(((second_fd - second).abs() / (1 + second.abs())).detach().max())
    assert torch.allclose(second_fd, second, atol=1e-3, rtol=2e-3), (
        "Incorrect radial second derivative"
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters, "The current ASYS fitting interface requires fitted parameters"
    gradients = torch.autograd.grad(model(points).square().sum(), parameters, allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    assert any(g is not None for g in gradients), "Field disconnected from every parameter"
    sensitivity = max(float(g.detach().abs().max()) for g in gradients if g is not None)
    if require_parameter_sensitivity:
        assert sensitivity > 1e-12, "Unfitted model has negligible parameter sensitivity"
    return {
        "passes": True,
        "batch_error": batch_error,
        "relative_derivative_errors": errors,
        "relative_radial_second_derivative_error": second_error,
        "maximum_squared_field_parameter_gradient": sensitivity,
        "parameter_sensitivity_required": require_parameter_sensitivity,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    config = yaml.safe_load((args.problem / "config.yaml").read_text())
    model = _build_model(
        _load_module(args.problem / "model.py", "candidate"), config, torch.device("cpu")
    )
    if args.state:
        model.load_state_dict(torch.load(args.state, map_location="cpu", weights_only=True))
    result = check(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
