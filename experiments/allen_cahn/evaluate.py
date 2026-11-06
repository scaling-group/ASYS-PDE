"""Assess frozen formulas and neural controls against their stated references."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from metrics import crossings, energy, interface, pool, predict, weights

ROOT = Path(__file__).resolve().parent


def load(source, state, config):
    spec = importlib.util.spec_from_file_location('ac_frozen_candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Candidate(config).cpu().double()
    model.load_state_dict(torch.load(state, map_location='cpu', weights_only=True), strict=True)
    return model.eval().requires_grad_(False)


def evaluate(original_reference=None, current_reference=None):
    rows = []
    for label, path, floor in [('original', original_reference, 1e-14), ('neural', current_reference, 1e-12)]:
        if path is None:
            continue
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata['initial_regularizer'] != floor:
            raise ValueError(f'{label} requires initial regularizer {floor:g}')
        with np.load(path, allow_pickle=False) as bank:
            axis, times, truth = bank['x'], bank['t'], bank['u']
        if truth.shape != (262, 384, 384) or len(times) != 262 or not np.isfinite(truth).all():
            raise ValueError('Expected the full 384-square, 262-time reference bank')
        expected_axis = -1 + 2*np.arange(384)/384
        if not np.array_equal(axis, expected_axis) or times[0] != 0 or times[-1] != 8:
            raise ValueError('Unexpected periodic grid or time window')
        time_weights = weights(times)
        contours = [crossings(v) for v in truth]
        cases = []
        for name in (('formula14', 'formula12') if label == 'original' else ('formula14',)):
            folder = ROOT / 'selected' / name
            cases.append((name, folder/'model.py', folder/'state_dict.pt',
                          yaml.safe_load((folder/'config.yaml').read_text())))
        if label == 'neural':
            cases.extend((str(seed), ROOT/'neural/sources/neural_template.py',
                          ROOT/f'neural/states/{seed}/selected.pt', {})
                         for seed in (2026110401, 2026110411, 2026110421))
        for name, source, state, config in cases:
            model = load(source, state, config)
            per_time = []
            for i, time in enumerate(times):
                values = predict(model, axis, float(time))
                if values.shape != truth[i].shape or not np.isfinite(values).all():
                    raise ValueError('Invalid frozen field')
                per_time.append(dict(time=float(time), weight=float(time_weights[i]),
                                     **energy(values, truth[i]),
                                     **interface(crossings(values), contours[i])))
            rows.append(dict(candidate=name, initial_regularizer=floor,
                             **pool(per_time), per_time=per_time))
    return dict(units='Field and band errors are percentages; interface distances use physical coordinates.', rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-reference', type=Path)
    parser.add_argument('--current-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.original_reference is None and args.current_reference is None:
        parser.error('Supply at least one reference bank')
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    result = evaluate(args.original_reference, args.current_reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
