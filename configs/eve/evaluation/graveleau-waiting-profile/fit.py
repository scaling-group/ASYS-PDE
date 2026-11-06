"""Bounded fitting; official fitness is computed by the fresh score interpreter."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def run(solver, output, seconds, judge):
    evaluation_start = time.monotonic()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    sys.path.insert(0,str(solver))
    import task
    from check import check_field
    output.mkdir(parents=True,exist_ok=False)
    hashes = {n:digest(solver/n) for n in ('model.py','loss.py','optim.py','config.yaml')}
    snapshot = output/'frozen_candidate'
    snapshot.mkdir()
    input_hashes = {n:digest(solver/n) for n in (*hashes,'task.py','check.py')}
    for name in input_hashes:
        shutil.copyfile(solver/name,snapshot/name)
    evaluator = Path(__file__).resolve().parent
    evaluator_hashes = {n:digest(evaluator/n) for n in
                        ('fit.py','score.py','qualify.py','evaluation.sh')}
    archive = output/'frozen_evaluator'
    archive.mkdir()
    for name in evaluator_hashes:
        shutil.copyfile(evaluator/name,archive/name)
    cfg = yaml.safe_load((solver/'config.yaml').read_text())
    torch.manual_seed(int(cfg.get('seed',0)))
    model = task.load_model(solver/'model.py')
    torch.save(model.state_dict(),output/'initial_state.pt')
    initial_checks = task.physical_checks(model)
    initial_derivatives = check_field(model)
    loss_fn = module(solver/'loss.py','fitting_loss').loss_fn
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = module(solver/'optim.py','fitting_optimizer').build_optimizer(model,cfg) if parameters else None
    if optimizer is not None and not isinstance(optimizer,torch.optim.Optimizer):
        raise TypeError('build_optimizer must return a torch optimizer')
    steps = int(os.environ.get('ASYS_GRAVELEAU_MAX_STEPS',cfg.get('max_steps',2000)))
    if not 0<=steps<=100000:
        raise ValueError('Invalid requested update count')
    batch = task.samples(241)
    best = valid = float('inf')
    best_state = valid_state = None
    best_step = valid_step = 0
    history = []
    calls = 0
    setup_seconds = time.monotonic()-evaluation_start
    start = time.monotonic()
    stop = 'update_limit'

    class TimeLimit(Exception):
        pass

    def closure():
        nonlocal calls
        if time.monotonic()-start >= seconds:
            raise TimeLimit
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        value = loss_fn(model,batch,task)
        if not isinstance(value,torch.Tensor) or value.numel()!=1 or not torch.isfinite(value):
            raise ValueError('Training loss must be one finite scalar tensor')
        if optimizer is not None:
            value.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError('Nonfinite fitting gradient')
        calls += 1
        return value

    completed = 0
    try:
        for step in range(steps+1):
            value = float(closure().detach())
            completed = step
            if value < best:
                best,best_state,best_step = value,copy.deepcopy(model.state_dict()),step
                torch.save(best_state,output/'best_loss_state.pt')
            if judge=='corrected' and (step%10==0 or step==steps) and value<valid:
                # Eligibility must describe the saved state, not mutable fitting helpers.
                torch.save(model.state_dict(),output/'eligibility_state.pt')
                result=subprocess.run([sys.executable,str(evaluator/'score.py'),
                    '--solver',str(snapshot),'--state',str(output/'eligibility_state.pt'),
                    '--eligibility-only'],check=True,capture_output=True,text=True)
                eligibility=json.loads(result.stdout)
                if eligibility['physical']['passes'] and eligibility['derivatives']['passes']:
                    valid,valid_state,valid_step = value,copy.deepcopy(model.state_dict()),step
                    torch.save(valid_state,output/'best_valid_state.pt')
            grad = max((float(p.grad.abs().max()) for p in parameters if p.grad is not None),default=0)
            row = {'step':step,'loss':value,'gradient_max':grad,'elapsed_seconds':time.monotonic()-start}
            history.append(row)
            with (output/'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            if grad<=1e-8:
                stop = 'gradient_convergence' if optimizer is not None else 'parameter_free'
                break
            if len(history)>=51:
                recent = [r['loss'] for r in history[-50:]]
                if max(recent)-min(recent)<=1e-12*max(1,abs(value)):
                    stop = 'loss_stagnation_50_updates'
                    break
            if step<steps:
                if isinstance(optimizer,torch.optim.LBFGS):
                    optimizer.step(closure)
                else:
                    optimizer.step()
    except TimeLimit:
        stop = 'cooperative_time_limit'
    fit_seconds = time.monotonic()-start
    use_valid = judge == "corrected" and valid_state is not None
    state = valid_state if use_valid else best_state
    if state is None:
        raise RuntimeError('No finite fitting state; initial state retained')
    torch.save(state,output/'selected_state.pt')
    report = {'candidate_sha256':hashes,'source_sha256':hashes['model.py'],
              'input_sha256':input_hashes,'protected_evaluator_sha256':evaluator_hashes,
              'state_sha256':{n:digest(output/n) for n in ('initial_state.pt','selected_state.pt')},
              'public_judge':judge,
              'fitting_initial_checks':initial_checks,'fitting_initial_derivatives':initial_derivatives,
              'parameters':sum(p.numel() for p in parameters),'requested_updates':steps,
              'completed_updates':completed,'selected_step':valid_step if use_valid else best_step,
              'stop_reason':stop,'closure_evaluations':calls,'optimization_timer_seconds':seconds,
              'fit_elapsed_seconds':fit_seconds,'setup_elapsed_seconds':setup_seconds,
              'fitting_evaluation_elapsed_seconds':time.monotonic()-evaluation_start,
              'selected_loss':valid if use_valid else best,
              'diagnostic_scope':'Fitting-time diagnostics and editable loss; no official public fitness'}
    if input_hashes != {n:digest(solver/n) for n in input_hashes}:
        raise RuntimeError('Candidate or protected task files changed during fitting')
    if evaluator_hashes != {n:digest(evaluator/n) for n in evaluator_hashes}:
        raise RuntimeError('Protected evaluator files changed during fitting')
    (output/'fit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=300)
    parser.add_argument('--judge',choices=['corrected'],default='corrected')
    args = parser.parse_args()
    if args.seconds<=0:
        raise ValueError('Fitting timer must be positive')
    print(json.dumps(run(args.solver.resolve(),args.output.resolve(),args.seconds,args.judge),allow_nan=False))
