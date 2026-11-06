"""Equal-snapshot Allen–Cahn field and interface readout."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import torch
import yaml


def field(model, x, y, time):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, time)))
    values = []
    with torch.no_grad():
        for part in np.array_split(points, max(1, (len(points) + 4095) // 4096)):
            values.append(model(torch.tensor(part, dtype=torch.float64)).numpy()[:, 0])
    return np.concatenate(values).reshape(xx.shape)


def crossings(u, x, y):
    # Linear edge crossings on an endpoint-free periodic Cartesian grid.
    points = []
    for axis, spacing in ((0, x[1] - x[0]), (1, y[1] - y[0])):
        v = np.roll(u, -1, axis=axis)
        rows, cols = np.nonzero((u >= 0) != (v >= 0))
        fraction = u[rows, cols] / (u[rows, cols] - v[rows, cols])
        p = np.column_stack((x[rows], y[cols]))
        p[:, axis] += fraction * spacing
        points.append((p + 1.0) % 2.0 - 1.0)
    return np.concatenate(points)


def interface_error(a, b):
    if not len(a) or not len(b):
        return {"edge_crossings_a": len(a), "edge_crossings_b": len(b),
                "symmetric_rms": None, "maximum": None}
    nearest = []
    for source, target in ((a, b), (b, a)):
        for part in np.array_split(source, max(1, (len(source) + 511) // 512)):
            delta = (part[:, None, :] - target[None, :, :] + 1.0) % 2.0 - 1.0
            nearest.extend(np.sqrt(np.square(delta).sum(axis=2).min(axis=1)))
    nearest = np.asarray(nearest)
    return {"edge_crossings_a": len(a), "edge_crossings_b": len(b),
            "symmetric_rms": float(np.sqrt(np.square(nearest).mean())),
            "maximum": float(nearest.max())}



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True, help='Saved baseline evaluation directory with source/, state.pt, effective_config.yaml')
    p.add_argument('--reference', type=Path, required=True, help='Twenty-snapshot Fourier-Strang NPZ')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    source = args.log / 'source'
    saved = json.loads((args.log / 'manifest.json').read_text())
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    assert saved['state_sha256'] == sha(args.log / 'state.pt')
    assert saved['evaluator_sha256'] == sha(source / 'evaluate.py')
    assert all(sha(source / name) == digest for name, digest in saved['source_sha256'].items())
    assert saved['effective_config_sha256'] == sha(args.log / 'effective_config.yaml')
    spec = importlib.util.spec_from_file_location('frozen_ac_evaluator', source / 'evaluate.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    config = yaml.safe_load((args.log / 'effective_config.yaml').read_text())
    _, _, model = evaluator.load(source, config)
    model.load_state_dict(torch.load(args.log / 'state.pt', weights_only=True, map_location='cpu'), strict=True)
    model.eval()
    problem = evaluator.module('_ac_assessment_problem', source / 'problem.py')
    replay = evaluator.public_score(problem, model)
    original = yaml.safe_load((args.log / 'score.yaml').read_text())
    assert all(abs(replay[k] - original[k]) <= 1e-12 * (1 + abs(original[k]))
               for k in ('score', 'physics', 'ic', 'ic_tangent', 'bc'))
    with np.load(args.reference, allow_pickle=False) as data:
        fine = {key: data[key] for key in ('x', 'y', 't', 'u')}
    expected_times = np.array([0., .42, .844, 1.264, 1.684, 2.104, 2.528, 2.948, 3.368,
        3.7880000000000003, 4.212, 4.632, 5.0520000000000005, 5.472, 5.896, 6.316,
        6.736, 7.156000000000001, 7.58, 8.])
    if not np.array_equal(fine['t'], expected_times):
        raise ValueError('Expected the twenty geometric-field snapshot times')
    if not (np.array_equal(fine['x'], -1 + 2 * np.arange(512) / 512)
            and np.array_equal(fine['x'], fine['y']) and fine['u'].shape == (20,512,512)
            and np.isfinite(fine['u']).all()):
        raise ValueError('Expected finite 20 by 512 by 512 periodic reference')
    rows, error_squared, norm_squared = [], [], []
    for time, target in zip(fine['t'], fine['u']):
        actual = field(model, fine['x'], fine['y'], time)
        if actual.shape != target.shape or not np.isfinite(actual).all():
            raise ValueError('Invalid frozen field output')
        e2, n2 = float(np.square(actual - target).sum()), float(np.square(target).sum())
        error_squared.append(e2); norm_squared.append(n2)
        rows.append(dict(t=float(time), relative_l2=float(np.sqrt(e2/n2)),
            interface=interface_error(crossings(actual, fine['x'], fine['y']), crossings(target, fine['x'], fine['y']))))
    rms = [r['interface']['symmetric_rms'] for r in rows]
    result = dict(relative_l2_equal_snapshot_weight=float(np.sqrt(sum(error_squared)/sum(norm_squared))),
        largest_snapshot_interface_rms=max(rms) if all(v is not None for v in rms) else None,
        snapshots=rows, relative_error_units='fraction')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
