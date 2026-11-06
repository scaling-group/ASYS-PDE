"""Frozen Gaussian continuations and Laguerre concentrating densities."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diagnostics import aggregate, h, temporal
from reference import density_nodes

ROOT = Path(__file__).resolve().parent


def load(name):
    folder = ROOT/'selected'/name
    spec = importlib.util.spec_from_file_location('ks_'+name, folder/'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Candidate(yaml.safe_load((folder/'config.yaml').read_text()))
    model.load_state_dict(torch.load(folder/'state_dict.pt', map_location='cpu', weights_only=True), strict=True)
    return model.eval()


def evaluate(reference=None, physics=False):
    rows = []
    if reference is not None:
        with np.load(reference, allow_pickle=False) as data:
            radius, times, mass = data['radius'], data['times'], data['mass']
        if len(radius) not in (96001,192001) or mass.shape != (len(radius)-2,77) or len(times) != 77:
            raise ValueError('Expected the original 96k/192k mass-reference geometry')
        stride = (len(radius)-1)//48000
        truth = density_nodes(radius, mass, stride)
        radius = radius[::stride]
    for name in ('gaussian','gaussian_continued','reweighted','laguerre4','laguerre48'):
        model = load(name)
        row = dict(candidate=name, parameters=sum(p.numel() for p in model.parameters()))
        if physics:
            points = temporal(model, h._monitor_times(), 1280)
            row['equation_residual'] = aggregate(points)
            row['per_time_residual'] = points
        if reference is not None:
            errors, norms, centers = [], [], []
            for time, target in zip(times, truth):
                points = torch.tensor(np.column_stack((radius, np.full(len(radius),time))))
                with torch.no_grad():
                    values = torch.cat([model(p)[:,0] for p in points.split(16384)]).numpy()
                if not np.isfinite(values).all():
                    raise ValueError('Non-finite frozen density')
                errors.append(float(np.trapezoid((values-target)**2*radius, radius)))
                norms.append(float(np.trapezoid(target**2*radius, radius)))
                centers.append(float(values[0]/target[0]-1))
            row['relative_space_time_radial_l2'] = float(np.sqrt(np.trapezoid(errors,times)/np.trapezoid(norms,times)))
            row['per_time_relative_radial_l2'] = np.sqrt(np.array(errors)/norms).tolist()
            row['per_time_relative_center_error'] = centers
        rows.append(row)
    return dict(units='Relative errors are fractions; field norm uses r dr dt.', rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--physics', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.reference is None and not args.physics:
        parser.error('Supply --reference or --physics')
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    result = evaluate(args.reference, args.physics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
