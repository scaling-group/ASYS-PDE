"""Radial Keller–Segel evaluation: fitting and fresh-state scoring are separate CLI phases."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from problem import DTYPE, PUBLIC_CELLS, KSHelpers, _build_model, _build_optimizer, _fit, _grad, _load_module, _scalar_field
from check import check


def identity_points(arm):
    radii = torch.tensor([0., 5e-6, .003, .075, .7, 2.3, 12.7, 30.], dtype=DTYPE)
    times = [.02, .11, .5, .83, .86652]
    if arm != 'no-tangent':
        times = [0., 1e-8, 1e-5, 1e-4] + times
    rr, tt = torch.meshgrid(radii, torch.tensor(times, dtype=DTYPE), indexing='ij')
    return torch.stack((rr.flatten(), tt.flatten()), dim=1)


def field_probe(model, arm):
    points = identity_points(arm)
    with torch.enable_grad():
        coords = points.detach().clone().requires_grad_(True)
        value = _scalar_field(model(coords))
        first = _grad(value, coords)
        second = _grad(first[:, :1], coords)[:, :1]
    radii = torch.unique(points[:, 0], sorted=True)
    ic_points = torch.stack((radii, torch.zeros_like(radii)), dim=1)
    with torch.no_grad():
        plain = _scalar_field(model(points))
        ic_value = _scalar_field(model(ic_points))
    tensors = {'points': points, 'value': value, 'value_no_grad': plain,
               'dr': first[:, :1], 'dt': first[:, 1:], 'drr': second,
               'ic_points': ic_points, 'ic_value': ic_value}
    if not all(bool(torch.isfinite(x).all()) for x in tensors.values()):
        raise ValueError('Nonfinite exported field probe')
    return {key: x.detach().cpu().numpy() for key, x in tensors.items()}


def hashes(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def public_scores(model, arm):
    # A fresh helper with no candidate settings protects the public equation/grid.
    helpers = KSHelpers({}, torch.device('cpu'))
    check(model, require_parameter_sensitivity=False)
    times = helpers._monitor_times()
    if arm == 'no-tangent':
        times = times[times > 0]
    losses = []
    for chunk in times.split(2):
        ut, diffusion, chemistry, values, measure = helpers.continuous_parts(model, PUBLIC_CELLS, chunk)
        if not bool(torch.isfinite(values).all()) or float(values.detach().min()) < -1e-10:
            raise ValueError('Invalid public density')
        residual = ut - diffusion - chemistry
        if arm == 'raw-mse':
            loss = (residual.square() * measure).sum(1) / measure.sum()
        else:
            def norm(field):
                return torch.sqrt((field.square() * measure).sum(1).clamp_min(1e-30))
            loss = norm(residual) / (norm(ut) + norm(diffusion) + norm(chemistry) + 1e-10)
        if not bool(torch.isfinite(loss).all()):
            raise ValueError('Nonfinite public physics')
        losses.append(loss.detach())
    physics_values = torch.cat(losses)
    physics = physics_values.mean() if arm == 'raw-mse' else torch.exp(torch.log(physics_values.clamp_min(1e-12)).mean())
    points, radii, _, _ = helpers._grid(256, 1)
    coords = points.detach().requires_grad_(True)
    values = _scalar_field(model(coords))
    target = helpers.initial_condition(radii)
    components = {'physics': physics, 'ic': (values[:, 0] - target).square().mean()}
    if arm != 'no-tangent':
        tangent = _grad(values, coords)[:, 1]
        rhs = (4 * radii.square() - 13.5) * target + 2 * target.square()
        components['ic_tangent'] = (tangent - rhs).square().mean()
    boundary_times = torch.linspace(0, helpers.T_MAX, 256, dtype=DTYPE)[:, None]
    components['bc'] = helpers.boundary_loss(model, boundary_times)
    scores = {key: -float(value.detach()) for key, value in components.items()}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError('Nonfinite public score')
    scores['score'] = sum(scores.values()) / len(scores)
    scores['summary'] = f'KS continuous-IVP {arm}; {len(times)} public times; verified fresh state; negative losses'
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    parser.add_argument('arm', choices=('raw-mse',))
    args = parser.parse_args()
    torch.set_default_dtype(DTYPE)
    torch.set_num_threads(max(1, int(os.environ.get('ASYS_NUM_THREADS', '4'))))
    solver = Path(os.environ['EVE_SOLVER_ROOT']).resolve()
    logs = Path(os.environ['EVE_EVAL_LOG_ROOT']).resolve()
    artifacts = logs / 'eval_artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    problem = solver / 'problems/blowup/ks_radial_2d'
    fixed = Path(__file__).resolve().parent
    sources = [problem / name for name in ('model.py', 'loss.py', 'optim.py', 'config.yaml')]
    fixed_sources = [fixed / name for name in ('evaluate.py', 'problem.py', 'check.py', 'evaluate_' + args.arm.replace('-', '_') + '.sh')]
    workspace_fixed = [solver / p.name for p in fixed_sources]
    bound_sources = list(dict.fromkeys([*sources, *fixed_sources, *workspace_fixed]))
    source_hashes = hashes(bound_sources)
    state = artifacts / 'state_dict.pt'
    probe_path = artifacts / 'fitted_field_probe.npz'
    identity_path = artifacts / 'export_identity.json'
    required = [*bound_sources, state, probe_path]
    if args.phase == 'score':
        identity = json.loads(identity_path.read_text())
        if identity['arm'] != args.arm or hashes(required) != identity['sha256']:
            raise RuntimeError('Fitted export is not bound to this task/source/state/arm')
    config = yaml.safe_load((problem / 'config.yaml').read_text()) or {}
    torch.manual_seed(int(config.get('seed', 0)))
    model = _build_model(_load_module(problem / 'model.py', 'ks_candidate_model'), config, torch.device('cpu'))
    if args.phase == 'fit':
        helpers = KSHelpers(config, torch.device('cpu'))
        batch = helpers.training_batch()
        loss = _load_module(problem / 'loss.py', 'ks_candidate_loss')
        optim = _load_module(problem / 'optim.py', 'ks_candidate_optim')
        optimizer = _build_optimizer(optim, model, config)
        history, best, steps = _fit(model, loss.loss_fn, optimizer, batch, helpers, float(os.environ['ASYS_WALL_TIME']))
        model.eval()
        torch.save(model.state_dict(), state)
        np.savez(artifacts / 'loss_history.npz', loss=np.asarray(history, dtype=np.float64))
        probe = field_probe(model, args.arm)
        np.savez(probe_path, **probe)
        if hashes(bound_sources) != source_hashes:
            raise RuntimeError('Candidate/fixed source changed during fitting')
        identity = {'arm': args.arm, 'sha256': hashes(required),
                    'probe_fields': list(probe), 'probe_points': len(probe['points']), 'rtol': 1e-10, 'atol': 1e-12}
        identity_path.write_text(json.dumps(identity, indent=2) + '\n')
        report = {'train_best_loss': best, 'steps': steps, 'parameters': sum(p.numel() for p in model.parameters()),
                  'live_score_used': False, 'wall_time_setting': float(os.environ['ASYS_WALL_TIME']),
                  'training_points': len(batch.collocation), 'training_initial_time_points': int((batch.collocation[:, 1] == 0).sum())}
        (logs / 'fitting_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, sort_keys=True))
        return
    model.load_state_dict(torch.load(state, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    cold = field_probe(model, args.arm)
    deltas = {}
    with np.load(probe_path, allow_pickle=False) as live:
        for key, values in cold.items():
            if live[key].shape != values.shape or not np.allclose(live[key], values, rtol=1e-10, atol=1e-12):
                raise RuntimeError('Fitted/cold field or derivative mismatch: ' + key)
            deltas[key] = float(np.max(np.abs(live[key] - values)))
    training_sources = {str((problem / name).resolve()) for name in ('loss.py', 'optim.py')}
    def training_imported():
        return any(getattr(module, '__file__', None) and str(Path(module.__file__).resolve()) in training_sources for module in list(sys.modules.values()))
    if training_imported():
        raise RuntimeError('Cold reconstruction imports candidate training code')
    with torch.enable_grad():
        scores = public_scores(model, args.arm)
    if training_imported() or hashes(required) != identity['sha256']:
        raise RuntimeError('Score imports training code or changes an input')
    provenance = {'fresh_process': True, 'arm': args.arm, 'candidate_training_code_imported': False,
                  'cold_probe_max_absolute_differences': deltas, 'sha256': hashes([*required, identity_path])}
    (logs / 'frozen_score_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (logs / 'score.yaml').write_text(json.dumps(scores, indent=2) + '\n')
    print(json.dumps(scores, sort_keys=True))


if __name__ == '__main__':
    main()
