"""Evaluate the prescribed C+B, C+B+P and C+B+P+E fields without fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metrics import breather, metrics, pool, predict
from model import Candidate

ROOT = Path(__file__).resolve().parent


def evaluate(reference_root, widths, orders):
    config = json.loads((ROOT / 'config.json').read_text())
    rows = []
    for order in orders:
        model = Candidate(dict(config, fixed_predictor_order=order)).eval()
        if torch.count_nonzero(model.coefficients):
            raise ValueError('Prescribed fields require zero learned deviations')
        for width in widths:
            with np.load(reference_root / f'L{width:g}/reference.npz', allow_pickle=False) as data:
                x, times, truth = data['readout_x'], data['readout_times'], data['readout_state']
                if float(data['width']) != width or truth.shape != (len(times), len(x)):
                    raise ValueError('Reference width or field shape mismatch')
            values = predict(model, x, times, width)
            rows.append(dict(order=order, width=width, nx=len(x), nt=len(times),
                             **metrics(values, truth, breather(x, times), times, width)))
    pooled = [dict(order=order, widths=list(widths),
                   **pool([row for row in rows if row['order'] == order])) for order in orders]
    return dict(units='Relative errors are fractions, not percentages.', rows=rows, pooled=pooled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--widths', type=float, nargs='+', choices=(3.5, 4., 4.5, 5., 6., 7.), default=(3.5, 4., 4.5, 5., 6., 7.))
    parser.add_argument('--orders', type=int, nargs='+', choices=(0, 1, 2), default=(0, 1, 2))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    result = evaluate(args.reference_root, args.widths, args.orders)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
