"""Assess H3, the nine search outcomes, or the four matched-coordinate states."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'scripts/references'))
from fixed_point_reference import CanonicalProfile, LogProfile


def tail_readout(raw, normalization, family="asys"):
    """Reuse exact 80-digit endpoint algebra for H3 and its matched controls."""
    sys.path.insert(0, str(ROOT / "matched_coordinates/source"))
    from endpoint import endpoint_algebra

    q0, q1, check = endpoint_algebra(raw, family)
    beta = float(raw.beta())
    alpha = 1 / beta
    amplitude = normalization['amplitude']
    c = math.exp(q0) * amplitude**(-alpha - 1)
    d = math.exp(q0) * (q1 - 2*q0) * amplitude**(-2*alpha - 1)
    lam = (1 + beta) / .75
    k = lam / math.tan(math.pi*alpha/2) * (.25*alpha/(alpha-1) - 1)
    return dict(A=amplitude, Q0=q0, Qprime0=q1, C=c, D=d, lambda_D=lam*d, K=k,
                tail_balance_error_percent=100*(d/(c*c*k)-1),
                endpoint_algebra_forward_max_abs=check)


def evaluate(reference_root, matched=False):
    source = ROOT / ('matched_coordinates/source' if matched else 'selected')
    for name in ('model', 'frozen_profile', 'panel_models', 'endpoint'):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(source))
    from model import build_model
    from frozen_profile import FrozenProfile

    arms = ('poly_identity', 'poly_gram', 'asys_identity', 'asys_gram') if matched else ('H3',)
    rows = []
    for arm in arms:
        os.environ['MATCHED_ARM'] = arm
        folder = ROOT / 'matched_coordinates/states' / arm if matched else source
        config = json.loads((folder / 'model_config.json').read_text())
        normalization = json.loads((folder / 'normalization.json').read_text())
        raw = build_model(config)
        model = FrozenProfile(raw, normalization['amplitude'], normalization['origin_slope'])
        model.load_state_dict(torch.load(folder / 'state_dict.pt', map_location='cpu', weights_only=True), strict=True)
        y = np.logspace(-3, 3, 257)
        with torch.no_grad():
            values = model(.25, torch.tensor(y)).numpy()
        if not np.isfinite(values).all():
            raise ValueError('Non-finite frozen profile')
        metrics = {}
        for name in ('coarse', 'medium', 'fine', 'extended_domain'):
            with np.load(reference_root / name / 'profile.npz', allow_pickle=False) as data:
                profile = LogProfile(data['w'], data['log_values'], float(data['tail_power']))
                truth = CanonicalProfile(profile, float(data['b'])).evaluate(y)[0]
            delta = values - truth
            metrics[name] = dict(
                profile_error_percent=100 * float(np.linalg.norm(delta) / np.linalg.norm(truth)),
                physical_radius_l2_error_percent=100 * float(np.sqrt(np.trapezoid(delta**2, y) / np.trapezoid(truth**2, y))),
                peak_normalized_max_error_percent=100 * float(np.max(abs(delta)) / np.max(abs(truth))),
            )
        row = dict(arm=arm, beta=float(raw.beta()), references=metrics)
        row.update(tail_readout(raw, normalization, arm.split('_')[0] if matched else "asys"))
        rows.append(row)
    return dict(units='Errors are percentages; tail balance is signed.', rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--matched', action='store_true')
    group.add_argument('--cohort', action='store_true')
    parser.add_argument('--public', action='store_true', help='Also run the fixed equation assessor for --cohort')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    if args.public and not args.cohort:
        parser.error('--public requires --cohort')
    if args.cohort:
        from cohort_readout import evaluate as evaluate_cohort
        result = evaluate_cohort(args.reference_root, args.public)
    else:
        result = evaluate(args.reference_root, args.matched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
