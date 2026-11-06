"""Frozen radial Fujita formulas, response swaps and neural controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from metrics import weights

ROOT = Path(__file__).resolve().parent


def formula(arm, state=None):
    folder = ROOT / 'selected' / arm
    spec = importlib.util.spec_from_file_location('fujita_' + arm, folder / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Candidate(yaml.safe_load((folder / 'config.yaml').read_text())).double()
    model.load_state_dict(torch.load(state or folder/'state_dict.pt', map_location='cpu', weights_only=True), strict=True)
    return model


def assess(model, radii, times, truth):
    rr, tt = np.meshgrid(radii, times)
    points = np.column_stack((rr.ravel(), tt.ravel()))
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        values = np.concatenate([model(torch.from_numpy(p)).numpy().ravel()
                                 for p in np.array_split(points, int(np.ceil(len(points)/8192)))]).reshape(truth.shape)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite frozen field')
    wt = weights(times)
    result = {}
    for label, wr in [('full', weights(radii)), ('core', weights(radii, .5))]:
        e = ((values-truth)**2) @ wr
        r = truth**2 @ wr
        if not np.isfinite(e).all() or not np.all(r > 0):
            raise ValueError('Invalid field energy')
        result[label+'_percent'] = 100*float(np.sqrt((wt@e)/(wt@r)))
        result[label+'_final_percent'] = 100*float(np.sqrt(e[-1]/r[-1]))
    return result


def residuals(model):
    """Original fitting samples and the manuscript's independent neural-control samples."""
    sys.path.insert(0, str(ROOT / 'neural/sources'))
    from fujita_problem import Problem

    problem = Problem('cpu')
    model.eval().requires_grad_(False)
    samples = {
        'training_uniform_normalized_pde': problem.training_batch(dict(Nf=512, NIC=64, NBC=32))[0],
        'training_core_normalized_pde': problem.core_points(192, 107015),
        'independent_uniform_normalized_pde': problem.batch(count=8192, seed=2026110596)[0],
        'independent_core_normalized_pde': problem.core_points(8192, 2026110597),
    }
    result = {}
    for name, points in samples.items():
        # The archived independent readout averages 256-point blocks by their sizes.
        blocks = points.split(256) if name.startswith('independent') else (points,)
        result[name] = sum(len(block) * float(problem.physics_loss(model, block).detach())
                           for block in blocks) / len(points)
    result['original_training_objective'] = (result['training_uniform_normalized_pde']
                                             + .2 * result['training_core_normalized_pde'])
    result['independent_objective'] = (result['independent_uniform_normalized_pde']
                                       + .2 * result['independent_core_normalized_pde'])
    if not np.isfinite(list(result.values())).all():
        raise ValueError('Non-finite frozen-state residual diagnostic')
    return result


def evaluate(reference, cases):
    with np.load(reference, allow_pickle=False) as data:
        radii, times, truth = data['r'], data['t'], data['u']
    if truth.shape != (289, 3549) or truth.shape != (len(times), len(radii)) or not np.isfinite(truth).all():
        raise ValueError('Expected the 289-time, 3549-radius Fujita reference')
    if radii[0] != 0 or radii[-1] != 5 or times[0] != 0 or times[-1] != .020385594:
        raise ValueError('Reference domain differs from the stated IVP')
    rows = []
    if 'selected' in cases:
        for arm in ('L', 'U'):
            rows.append(dict(case=arm, **assess(formula(arm), radii, times, truth)))
    if 'response-swap' in cases:
        for state in ('initial', 'L-fit', 'U-fit'):
            for arm in ('L', 'U'):
                model = formula(arm, ROOT/'response_swap'/f'{state}.pt')
                rows.append(dict(case='response-swap', state=state, response=arm,
                                 **residuals(model), **assess(model, radii, times, truth)))
    for sampling in ('dense', 'sparse'):
        if sampling not in cases:
            continue
        sys.path.insert(0, str(ROOT / 'neural'))
        from neural import Neural
        for seed in (2026110501, 2026110511, 2026110521):
            for budget, name in ((600, 'checkpoint600.pt'), (3600, 'selected.pt')):
                model = Neural(seed).double()
                model.load_state_dict(torch.load(ROOT/f'neural/states/{sampling}/{seed}'/name,
                                                  map_location='cpu', weights_only=True), strict=True)
                rows.append(dict(case=sampling, seed=seed, budget_seconds=budget,
                                 **assess(model, radii, times, truth)))
    return dict(units='Field errors are percentages; radial measure is dr, core radius is 0.5. Residual diagnostics are normalized MSEs.', rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=('selected','dense','sparse','response-swap'),
                        default=('selected','dense','sparse','response-swap'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    result = evaluate(args.reference, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
