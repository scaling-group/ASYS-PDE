"""Prospective IC resolution ties; raw public scores are retained verbatim."""
from __future__ import annotations

import math
import sys

POLICY = 'binary64-public-ic-rms-one-epsilon-v1'


def comparison_scores(raw, rows):
    if not rows:
        raise ValueError('IC comparison requires public rows')
    for row in rows:
        for key in ('ic', 'ic_target_energy', 'ic_epsilon'):
            value = row[key]
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid {key}')
        if row['ic_epsilon'] != sys.float_info.epsilon:
            raise ValueError('IC resolution policy requires binary64 reference precision')
        if not isinstance(row['ic_imag_nonzero'], bool):
            raise TypeError('IC imaginary flag must be boolean')
    worst_ic = max(row['ic'] for row in rows)
    if raw['ic'] != -worst_ic:
        raise ValueError('raw IC does not match the public rows')
    epsilon = sys.float_info.epsilon
    energy = max(row['ic_target_energy'] for row in rows)
    floor = epsilon**2 * energy
    imaginary_nonzero = any(row['ic_imag_nonzero'] for row in rows)
    resolution_tie = not imaginary_nonzero and worst_ic <= floor
    value = 0.0 if resolution_tie else raw['ic']
    return {**raw, 'ic_for_selection': value}, dict(
        policy=POLICY, epsilon=epsilon, reference_energy=energy,
        squared_rms_resolution=floor, raw_ic=raw['ic'],
        ic_for_selection=value, resolution_tie=resolution_tie,
        imaginary_nonzero=imaginary_nonzero, public_rows=len(rows),
    )
