"""Frozen baseline profile readout."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import torch
import yaml


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def reference(path):
    state = np.load(path / "profile.npz")
    profile = LogProfile(state["w"], state["log_values"], float(state["tail_power"]))
    return CanonicalProfile(profile, float(state["b"])), float(state["beta"])


def errors(values, truth, y):
    delta = values - truth
    return dict(
        log_grid_relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(truth)),
        physical_radius_relative_l2=float(np.sqrt(np.trapezoid(delta**2, y) / np.trapezoid(truth**2, y))),
        peak_normalized_maximum_error=float(np.max(abs(delta)) / np.max(abs(truth))),
    )



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('original-sigs','tail-sigs','original-fex','tail-fex'), required=True)
    p.add_argument('--candidate', type=Path, required=True, help='SIGS saved/ directory or FEX terminal candidate directory')
    p.add_argument('--references', type=Path, required=True, help='Directory containing fine/coarse/medium/extended_domain')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'scripts/references'))
    global CanonicalProfile, LogProfile
    from fixed_point_reference import CanonicalProfile, LogProfile
    variant, family = args.family.split('-')
    source = root / 'baseline/gclm' / ('original' if variant == 'original' else 'tail_capable') / family
    folder = args.candidate
    if family == 'sigs':
        sys.path[:0] = [str(source), str(root / 'baseline/gclm/kernel')]
        from single_anchor import load_profile
        model = load_profile(folder).eval()
        beta = float(model.beta(.25))
        evaluate = lambda points: model(.25, points)
    else:
        raw_module = module(source / 'model.py', 'readout_fex_model')
        if variant == 'original':
            raw = raw_module.Model(yaml.safe_load((folder / 'model.yaml').read_text())).double()
            raw.load_state_dict(torch.load(folder / 'state_dict.pt', map_location='cpu', weights_only=True), strict=True)
            checks = json.loads((folder / 'public.json').read_text())['profile_checks']
            amplitude = checks['normalization_amplitude']
            coordinate_scale = checks['normalization_coordinate_scale']
            if not np.isfinite([amplitude, coordinate_scale]).all() or min(amplitude, coordinate_scale) <= 0:
                raise ValueError('Positive finite saved normalization required')
            model = raw.eval()
            beta = float(raw.beta().detach())
            evaluate = lambda points: raw(coordinate_scale * points) / amplitude
        else:
            raw = raw_module.Model(json.loads((folder / 'model_config.json').read_text())).double()
            normal = json.loads((folder / 'normalization.json').read_text())
            frozen = module(source / 'frozen_profile.py', 'readout_fex_frozen')
            model = frozen.FrozenProfile(raw, normal['amplitude'], normal['origin_slope'])
            model.load_state_dict(torch.load(folder / 'frozen_state_dict.pt', map_location='cpu', weights_only=True), strict=True)
            if float(model.amplitude) != normal['amplitude'] or float(model.coordinate_scale) != normal['coordinate_scale']:
                raise ValueError('Saved normalization metadata/state disagreement')
            model.eval()
            beta = float(model.beta(.25))
            evaluate = lambda points: model(.25, points)
    y = np.logspace(-3, 3, 257)
    with torch.no_grad():
        values = evaluate(torch.tensor(y)).numpy()
    if values.shape != y.shape or not np.isfinite(values).all() or not np.isfinite(beta):
        raise ValueError('Invalid frozen profile output')
    metrics = {}
    for name in ('fine', 'coarse', 'medium', 'extended_domain'):
        truth, b = reference(args.references / name)
        metrics[name] = dict(**errors(values, truth.evaluate(y)[0], y), absolute_beta_error=abs(beta-b))
    print(json.dumps(dict(beta=beta, reference_metrics=metrics, relative_error_units='fraction'), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
