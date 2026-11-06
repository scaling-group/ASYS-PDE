"""Allen-Cahn neural control: fixed equation-based fitting and completed-step selection."""
import time
PROCESS_STARTED = time.monotonic()
PROCESS_CPU_STARTED = time.process_time()
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import torch
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parent
EPS = .04
ARMS = ('nn-lbfgs',)
PRODUCTION_SEEDS = (2026110401, 2026110411, 2026110421)

def seed_record(base):
    return dict(base_seed=base, initialization_seed=base+1000000,
        component_seeds=dict(residual=base, initial=base+1, boundary=base+2))

class BudgetStop(Exception): pass
class PhaseStop(Exception): pass


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(arm, seed, counts=(4096, 512, 64)):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed_record(seed)['initialization_seed'])
    path = ROOT / 'sources/official14'
    sys.path.insert(0, str(path))
    problem = load(path/'problem.py', '_current_ac_problem').Problem('cpu')
    candidate = path/'model.py' if arm == 'formula-lbfgs' else ROOT/'sources/neural_template.py'
    model = load(candidate, '_ac_candidate').Candidate({}).cpu().double()
    batch = problem.batch(count=counts[0], ic_count=counts[1], boundary_count=counts[2], seed=seed)
    return problem, model, batch


def objective(problem, model, batch):
    values = problem.loss_terms(model, *batch)
    focused = batch[0].clone()
    focused[:, :2] *= .55
    return 1000 * (sum(values.values()) + .35 * problem.physics_loss(model, focused))


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def batch_hash(batch):
    h = hashlib.sha256()
    for x in batch:
        h.update(x.detach().numpy().tobytes())
    return h.hexdigest()


def lbfgs(model):
    return torch.optim.LBFGS(model.parameters(), lr=.5, max_iter=1, max_eval=25,
        history_size=20, tolerance_grad=1e-9, tolerance_change=1e-12,
        line_search_fn='strong_wolfe')


def run(arm, seed, out, seconds=1200., counts=(4096,512,64), max_steps=100000,
        max_closures=20000, reserve=15., started=None, fault=None, adam_seconds=400., adam_updates=4000):
    """fault is validation-only; production CLI cannot inject a fault."""
    if arm not in ARMS or seconds <= reserve or reserve <= 0:
        raise ValueError('Invalid declared arm/budget')
    start = time.monotonic() if started is None else started
    cpu = time.process_time()
    deadline = start + seconds
    numerical_deadline = deadline - reserve
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    problem, model, batch = setup(arm, seed, counts)
    initial = clone_state(model)
    accepted = clone_state(model)
    best_state = clone_state(model)
    best = math.inf
    stop, failure = 'fit_time', None
    closures = completed_closures = accepted_steps = post_evaluations = outer_attempts = 0
    phase_counts = {}
    history = []
    at300 = None
    source_paths = [Path(__file__), ROOT/'sources/neural_template.py',
        ROOT/'sources/official14/model.py', ROOT/'sources/official14/problem.py',
        ROOT/'sources/official14/asys_task/problem.py',ROOT/'sources/official14/asys_task/operators.py']
    sources = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}

    def guard(phase_end=None):
        if time.monotonic() >= numerical_deadline:
            raise BudgetStop('fit_time_including_save_reserve')
        if phase_end is not None and time.monotonic() >= phase_end:
            raise PhaseStop('adam_phase_time')

    try:
        guard()
        initial_loss = float(objective(problem, model, batch).detach())
        post_evaluations += 1
        if not math.isfinite(initial_loss):
            raise FloatingPointError('Nonfinite initial loss')
        best = initial_loss
        stages = ('adam','lbfgs') if arm == 'nn-hybrid' else ('lbfgs',)
        for phase in stages:
            optimizer = torch.optim.Adam(model.parameters(), lr=.001) if phase == 'adam' else lbfgs(model)
            phase_end = min(start+adam_seconds, numerical_deadline) if phase == 'adam' else None
            phase_steps = 0
            phase_counts[phase] = 0
            while True:
                guard()
                if accepted_steps >= max_steps: raise BudgetStop('outer_step_cap')
                if closures >= max_closures: raise BudgetStop('closure_cap')
                if phase == 'adam' and (phase_steps >= adam_updates or time.monotonic() >= phase_end):
                    break
                before = clone_state(model)
                if phase == 'adam':
                    rate = .0001 + .0009*.5*(1+math.cos(math.pi*phase_steps/adam_updates))
                    optimizer.param_groups[0]['lr'] = rate
                def closure():
                    nonlocal closures, completed_closures
                    guard(phase_end)
                    if closures >= max_closures: raise BudgetStop('closure_cap')
                    closures += 1  # All attempted calls, including rejected/nonfinite trials.
                    optimizer.zero_grad(set_to_none=True)
                    value = objective(problem, model, batch)
                    if fault is not None:
                        fault(model, closures)  # Only used by allocated rollback tests.
                    if value.ndim or not bool(torch.isfinite(value)):
                        raise FloatingPointError('Nonfinite closure objective')
                    value.backward()
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise FloatingPointError('Nonfinite closure gradient')
                    completed_closures += 1
                    history.append(dict(kind='closure',phase=phase,step=accepted_steps+1,
                        closure=closures,seconds=time.monotonic()-start,loss=float(value.detach())))
                    guard(phase_end)
                    return value
                try:
                    outer_attempts += 1
                    optimizer.step(closure)
                    guard(phase_end)
                    optimizer.zero_grad(set_to_none=True)
                    value = float(objective(problem, model, batch).detach())
                    post_evaluations += 1
                    if not math.isfinite(value): raise FloatingPointError('Nonfinite post-step loss')
                    guard(phase_end)
                except PhaseStop:
                    model.load_state_dict(before,strict=True)
                    accepted = before
                    break  # Adam phase exhausted; discard optimizer and enter LBFGS.
                except BaseException:
                    model.load_state_dict(before,strict=True)
                    accepted = before
                    raise  # Budget/nonfinite/line-search failures stop entire fit; no retry.
                accepted_steps += 1
                phase_steps += 1
                phase_counts[phase] += 1
                accepted = clone_state(model)
                if value < best:
                    best, best_state = value, clone_state(model)
                history.append(dict(kind='accepted',phase=phase,step=accepted_steps,
                    seconds=time.monotonic()-start,loss=value))
                if at300 is None and time.monotonic()-start >= 300:
                    at300 = dict(state=copy.deepcopy(best_state),loss=best,
                        seconds=time.monotonic()-start,definition='First completed step at/after300s; actual elapsed recorded')
        stop = 'stage_caps'
    except BudgetStop as exc:
        stop = str(exc)
    except (FloatingPointError, ValueError, RuntimeError, OverflowError) as exc:
        stop, failure = 'failed', type(exc).__name__+': '+str(exc)
    # Saving and source checks consume the same overall fit budget.
    save_start = time.monotonic()
    model.load_state_dict(best_state,strict=True)
    torch.save(initial, out/'initial.pt')
    torch.save(accepted, out/'last_complete.pt')
    torch.save(best_state, out/'selected.pt')
    if at300 is not None: torch.save(at300,out/'checkpoint300.pt')
    (out/'history.json').write_text(json.dumps(history,separators=(',',':'))+'\n')
    for name,expected in sources.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            failure, stop = 'Scientific source changed during fitting','failed'
    artifact_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    elapsed = time.monotonic()-start
    result = dict(arm=arm,seed=seed,**seed_record(seed),counts=list(counts),batch_sha256=batch_hash(batch),
        source_sha256=sources,artifact_sha256=artifact_hashes,parameters=sum(p.numel() for p in model.parameters()),
        dtype=str(next(model.parameters()).dtype),device='cpu',threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        best_loss=best if math.isfinite(best) else None,accepted_steps=accepted_steps,outer_attempts=outer_attempts,
        closures_attempted=closures,closures_completed=completed_closures,
        post_step_objective_evaluations=post_evaluations,phase_steps=phase_counts,
        stop=stop,failure=failure,eligible=failure is None and accepted_steps>0,
        budget_seconds=seconds,save_reserve_seconds=reserve,
        elapsed_including_setup_sampling_posteval_and_saving=elapsed,
        cpu_seconds=time.process_time()-(PROCESS_CPU_STARTED if started is not None else cpu),save_seconds=time.monotonic()-save_start,
        budget_overrun=elapsed>seconds,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        torch_version=str(torch.__version__),reference_access=False)
    # Account final metadata serialization too; unavoidable uninterruptible overruns are failures.
    path=out/'fit.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    result['elapsed_including_setup_sampling_posteval_and_saving']=time.monotonic()-start
    result['budget_overrun']=time.monotonic()>deadline
    if result['budget_overrun']:
        result.update(eligible=False,stop='budget_overrun',failure='Budget includes final writes; no silent overrun acceptance')
    path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise SystemExit('Output already exists; no overwrite/retry of recorded repeat')
    try:
        result=run(args.arm,args.seed,args.out,started=PROCESS_STARTED)
    except Exception as exc:
        args.out.mkdir(parents=True,exist_ok=True)
        result=dict(arm=args.arm,seed=args.seed,eligible=False,stop='failed',
            failure=type(exc).__name__+': '+str(exc),reference_access=False,
            elapsed_including_setup_sampling_posteval_and_saving=time.monotonic()-PROCESS_STARTED)
        (args.out/'fit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['failure']: raise SystemExit(1)

if __name__=='__main__': main()
