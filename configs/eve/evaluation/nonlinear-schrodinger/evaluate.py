"""Native NLS fit/export and fresh, protected public assessment; no references."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
FIELDS = ('physics', 'ic', 'ic_tangent', 'bc')
FAILURE = {key: -1e9 for key in ('score', *FIELDS, 'ic_for_selection')}
PROBES = [[-1.13, .21, 4.], [.37, .47, 5.], [1.19, .83, 6.]]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hashes(application):
    return {str(path.relative_to(application)): digest(path)
            for path in sorted(application.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts}


def finite_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(v) for v in value)
    if isinstance(value, (float, complex)):
        return math.isfinite(value.real) and math.isfinite(value.imag)
    return isinstance(value, (str, int, bool, type(None)))


def same_state(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same_state(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def model_from_source(application, config):
    import torch
    sys.path.insert(0, str(application.resolve()))
    from asys_fit import _build_model, _load_module
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(int(config.get('seed', 0)))
    module = _load_module(application / 'model.py', 'candidate_model')
    return _build_model(module, config, torch.device('cpu'))


def probes(model):
    import torch
    from asys_fit import _grad
    points = torch.tensor(PROBES, dtype=torch.float64, requires_grad=True)
    values = model(points)
    if values.shape != (3, 2):
        raise ValueError('Invalid ordinary-probe shape')
    gradients = torch.stack([_grad(values[:, column:column + 1], points)
                             for column in (0, 1)], dim=1)
    if not torch.isfinite(values).all() or not torch.isfinite(gradients).all():
        raise ValueError('Nonfinite ordinary probes')
    return dict(points=PROBES, values=values.detach().tolist(),
                gradients=gradients.detach().tolist())


def fit_export(export, output, seed):
    """Training-only child. Its model/helper objects never reach assessment."""
    import torch
    import yaml
    application = export / 'application'
    started = time.monotonic()
    receipt = dict(model_calls=0, reference_access=False, fit_envelope_seconds=150,
                   maximum_fit_seconds=120, seed_override=seed)
    config = yaml.safe_load((application / 'config.yaml').read_text()) or {}
    if not isinstance(config, dict):
        raise TypeError('config.yaml must contain a mapping')
    if seed is not None:
        config['seed'] = seed
    model = model_from_source(application, config)
    from asys_fit import _build_optimizer, _fit, _load_module
    from problem import Helpers
    optimizer = _build_optimizer(_load_module(application / 'optim.py', 'candidate_optimizer'), model, config)
    loss_fn = _load_module(application / 'loss.py', 'candidate_loss').loss_fn
    helpers = Helpers(config, torch.device('cpu'))
    batch = helpers.training_batch()
    os.environ['ASYS_EARLYSTOP_PATIENCE_FRAC'] = '0.3'

    def checked_loss(*args):
        loss = loss_fn(*args)
        if not torch.is_tensor(loss) or loss.numel() != 1 or not torch.isfinite(loss).all():
            raise FloatingPointError('candidate fitting loss must be a finite scalar tensor')
        return loss

    fit_started = time.monotonic()
    history, best, steps = _fit(model, checked_loss, optimizer, batch, helpers, wall_time=150)
    receipt.update(fit_seconds=time.monotonic() - fit_started, fit_steps=steps, closure_calls=len(history))
    if not history:
        raise RuntimeError('fit returned no loss evaluation')
    model.eval()
    state = model.state_dict()
    if not finite_state(state):
        raise FloatingPointError('Nonfinite or unsupported exported state')
    torch.save(state, export / 'state_dict.pt')
    write(export / 'config.json', config)
    write(export / 'live_probes.json', probes(model))
    saved = torch.load(export / 'state_dict.pt', map_location='cpu', weights_only=True)
    if not same_state(saved, model.state_dict()):
        raise RuntimeError('Retained state changed during export probes')
    write(output / 'fit.json', dict(loss=best, history=history, steps=steps,
          parameters=sum(p.numel() for p in model.parameters()), buffers=sum(b.numel() for b in model.buffers())))
    receipt.update(state='exported', elapsed_seconds=time.monotonic() - started)
    write(output / 'fit_receipt.json', receipt)


def assess_export(export, result):
    """Cold child: imports model and immutable mathematics, never loss/optimizer."""
    import torch
    application = export / 'application'
    config = json.loads((export / 'config.json').read_text())
    model = model_from_source(application, config)
    saved = torch.load(export / 'state_dict.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved, strict=True)
    model.eval()
    if not finite_state(saved) or not same_state(saved, model.state_dict()):
        raise RuntimeError('Cold reconstruction did not restore exact exported state')
    from asys_fit import _load_module
    # Load the protected operator after candidate construction, into a fresh
    # namespace rather than any problem module imported by the model.
    problem = _load_module(application / 'problem.py', 'protected_public_problem')
    check_model = _load_module(application / 'check.py', 'protected_public_check').check_model
    scorer = problem.Helpers(config, torch.device('cpu')).score
    check = check_model(model)
    ordinary = probes(model)
    score, rows = scorer(model)
    comparison_scores = _load_module(Path(__file__).resolve().with_name('ic_comparison.py'),
                                     'protected_ic_comparison').comparison_scores
    score, comparison = comparison_scores(score, rows)
    if not same_state(saved, model.state_dict()):
        raise RuntimeError('Exported state changed during protected assessment')
    write(result, dict(score=score, rows=rows, comparison=comparison, check=check,
                      probes=ordinary, fits=0, model_calls=0, reference_access=False))


def assert_probe_equivalence(live, cold):
    if live['points'] != cold['points']:
        raise RuntimeError('Replay probe coordinates changed')

    def equivalent(left, right, rtol, atol):
        if isinstance(left, list):
            return isinstance(right, list) and len(left) == len(right) and all(
                equivalent(a, b, rtol, atol) for a, b in zip(left, right))
        return isinstance(right, (int, float)) and math.isfinite(left) and math.isfinite(right) and abs(
            left - right) <= atol + rtol * abs(right)

    for key, rtol, atol in [('values', 1e-10, 1e-11), ('gradients', 1e-9, 1e-10)]:
        if not equivalent(live[key], cold[key], rtol, atol):
            raise RuntimeError(f'Live/export ordinary {key} mismatch')


def run(application, output, seed=None, replay_from=None):
    """Protected parent: no candidate code is imported in this process."""
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('score.yaml', 'state_dict.pt', 'error.txt', 'ic_comparison.json'):
        (output / name).unlink(missing_ok=True)
    export = output / 'export'
    receipt = dict(model_calls=0, reference_access=False, seed_override=seed,
                   value_rtol=1e-10, value_atol=1e-11, gradient_rtol=1e-9, gradient_atol=1e-10)
    try:
        protected = {path: digest(path) for path in (Path(__file__).resolve(),
                     Path(__file__).resolve().with_name('ic_comparison.py'))}
        original = source_hashes(application)
        shutil.copytree(application, export / 'application', ignore=shutil.ignore_patterns('__pycache__'))
        if source_hashes(export / 'application') != original:
            raise RuntimeError('Application export differs from native source')
        executable = [sys.executable, str(Path(__file__).resolve())]
        common = ['--export', str(export), '--output', str(output)]
        if replay_from is None:
            command = executable + ['--fit-export'] + common
            if seed is not None:
                command += ['--seed', str(seed)]
            subprocess.run(command, check=True)
        else:
            shutil.copyfile(replay_from / 'state_dict.pt', export / 'state_dict.pt')
            shutil.copyfile(replay_from / 'config.json', export / 'config.json')
        if source_hashes(application) != original or source_hashes(export / 'application') != original:
            raise RuntimeError('Candidate source changed during fitting/export')
        bindings = {name: digest(export / name) for name in ('state_dict.pt', 'config.json')}
        if replay_from is None:
            bindings['live_probes.json'] = digest(export / 'live_probes.json')
        write(export / 'manifest.json', dict(source_hashes=original, export_hashes=bindings,
              protected_source_hashes={str(path): value for path, value in protected.items()}))

        def unchanged():
            if source_hashes(export / 'application') != original or any(
                    digest(export / name) != expected for name, expected in bindings.items()) or any(
                    digest(path) != expected for path, expected in protected.items()):
                raise RuntimeError('Frozen source/config/state/probes changed during assessment')

        results = []
        for index in (1, 2):
            unchanged()
            result = output / f'assessment_{index}.json'
            subprocess.run(executable + ['--assess-export', '--result', str(result)] + common, check=True)
            unchanged()
            results.append(json.loads(result.read_text()))
        first, second = results
        if any(first[key] != second[key] for key in ('score', 'rows', 'comparison')):
            raise RuntimeError('Repeated cold public assessments disagree')
        assert_probe_equivalence(first['probes'], second['probes'])
        if replay_from is None:
            live = json.loads((export / 'live_probes.json').read_text())
            for result in results:
                assert_probe_equivalence(live, result['probes'])
            receipt['live_export_equivalence'] = 'passed'
        else:
            previous = json.loads((replay_from / 'score.yaml').read_text())
            receipt['previous_score_difference'] = {
                key: first['score'][key] - previous[key] for key in (*FIELDS, 'score', 'ic_for_selection')}
            receipt['live_export_equivalence'] = 'unavailable: live probes were not recorded for the supplied state'
        shutil.copyfile(export / 'state_dict.pt', output / 'state_dict.pt')
        shutil.copyfile(export / 'config.json', output / 'config.json')
        write(output / 'check.json', first['check'])
        write(output / 'public_rows.json', first['rows'])
        write(output / 'ic_comparison.json', first['comparison'])
        write(output / 'score.yaml', {**first['score'], 'summary': 'Fixed periodic weak NLS public evaluation'})
        receipt.update(state='returned', cold_assessments=2, frozen_bindings_unchanged=True)
    except Exception as error:
        receipt.update(state='failed', error=f'{type(error).__name__}: {error}')
        (output / 'error.txt').write_text(traceback.format_exc())
        write(output / 'score.yaml', {**FAILURE, 'summary': receipt['error']})
    receipt['elapsed_seconds'] = time.monotonic() - started
    write(output / 'replay_receipt.json', receipt)
    return 0 if receipt['state'] == 'returned' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--application', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--fit-export', action='store_true')
    parser.add_argument('--assess-export', action='store_true')
    parser.add_argument('--result', type=Path)
    parser.add_argument('--replay-from', type=Path)
    args = parser.parse_args()
    if args.fit_export:
        started = time.monotonic()
        try:
            fit_export(args.export, args.output, args.seed)
        except Exception as error:
            write(args.output / 'fit_receipt.json', dict(state='failed',
                  error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started,
                  model_calls=0, reference_access=False, fit_envelope_seconds=150,
                  maximum_fit_seconds=120, seed_override=args.seed))
            raise
    elif args.assess_export:
        assess_export(args.export, args.result)
    else:
        if args.application is None:
            parser.error('--application is required')
        raise SystemExit(run(args.application, args.output, args.seed, args.replay_from))
