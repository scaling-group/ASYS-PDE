"""Fujita neural control with fixed dense or sparse collocation and completed-step selection."""
import os,time
START=time.monotonic();CPU=time.process_time()
from pathlib import Path
import sys,json,math,hashlib,argparse
from dataclasses import dataclass
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sources'))
import torch
from neural import Neural,TF,TC
from fujita_problem import Problem
from dense_sampling import dense_batch,verify_initial
SEEDS=(2026110501,2026110511,2026110521)
@dataclass(frozen=True)
class Settings:
    total:float=3600.
    train_end:float=3585.
    adam_end:float=2400.
    checkpoint:float=600.
    max_outer:int=100000
    max_closures:int=300000
    unchanged_stop:int=50
PROD=Settings()
class Budget(Exception):pass
class AdamEnd(Exception):pass

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def state(m):return {k:v.detach().clone() for k,v in m.state_dict().items()}
def physics_diagnostics(p,m,batch):
    d={}
    for label,points in [('uniform',p.batch(count=8192,seed=2026110596)[0]),('core',p.core_points(8192,2026110597))]:
        total=0.
        for b in points.split(256):total+=len(b)*float(p.physics_loss(m,b).detach())
        d[label]=dict(count=len(points),mse=total/len(points))
    ic=batch[1];ic=ic[ic[:,0]<5].clone().requires_grad_(True);u=m(ic);dt=torch.autograd.grad(u.sum(),ic)[0][:,1:2];rhs=p.initial_rhs(ic)
    d['interior_initial_tangent']=dict(mse=float((dt-rhs).square().mean().detach()),normalized_mse=float(((dt-rhs)/(rhs.abs()+1)).square().mean().detach()),wall_corner_excluded=True)
    return d

def _fit_claimed(seed,out,cfg=PROD,start=None,sampling="dense"):
    cpu=CPU if start is None else time.process_time();start=START if start is None else start
    assert 0<cfg.adam_end<cfg.train_end<cfg.total and 0<cfg.checkpoint<cfg.total
    out=out.resolve();assert out!=ROOT and ROOT not in out.parents,'Outputs must be outside immutable source tree'
    assert out.is_dir(),'Caller must atomically claim a fresh output directory'
    sources=json.loads((ROOT/'SOURCE_FREEZE.json').read_text())['sha256']
    for n,h in sources.items():assert sha(ROOT/n)==h,n
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    p=Problem('cpu');m=Neural(seed);assert sum(v.numel() for v in m.parameters())==12739
    paired_initial_sha256=verify_initial(m,seed)
    batch,core,sampling_density=dense_batch(p, blocks=8 if sampling=="dense" else 1)
    batch_hash={str(i):hashlib.sha256(v.numpy().tobytes()).hexdigest() for i,v in enumerate(batch)};batch_hash['core']=hashlib.sha256(core.numpy().tobytes()).hexdigest()
    initial=state(m);torch.save(initial,out/'initial.pt')
    write(out/'PRE_FIT_FREEZE.json',dict(seed=seed,initialization_seed=seed,training_seed=107014,core_seed=107015,batch_sha256=batch_hash,source_sha256=sources,settings=cfg.__dict__,sampling_density=sampling_density,paired_initial_sha256=paired_initial_sha256,reference_access=False))
    elapsed=lambda:time.monotonic()-start
    def objective():return p.physics_loss(m,batch[0])+.2*p.physics_loss(m,core)
    best=float(objective().detach());assert math.isfinite(best)
    measured=elapsed();selected=state(m);completed_time=elapsed();selected_step=0;selected_measured=measured;selected_completed=completed_time
    write(out/'initial-measurement.json',dict(loss=best,first_measured_elapsed=measured,completed_elapsed=completed_time))
    checkpoint_state=selected if completed_time<=cfg.checkpoint else None
    checkpoint_best=best if checkpoint_state is not None else float('inf')
    cp_step=0;cp_measured=measured;cp_completed=completed_time;cp_saved=False
    history=[];events=[];attempted=completed=accepted=post=outer=unchanged=0
    failure=None;stop='training_time';phase='adam';opt=torch.optim.Adam(m.parameters(),lr=.001)
    def save_checkpoint(force=False):
        nonlocal cp_saved
        if cp_saved or (not force and elapsed()<cfg.checkpoint):return
        if checkpoint_state is not None:
            torch.save(checkpoint_state,out/'checkpoint600.pt')
            write(out/'checkpoint600.json',dict(nominal_budget_seconds=cfg.checkpoint,best_loss=checkpoint_best,selected_outer_step=cp_step,first_measured_elapsed=cp_measured,completed_elapsed=cp_completed,write_elapsed=elapsed(),budget_reached=elapsed()>=cfg.checkpoint,strict_no_post_budget_state=True))
        cp_saved=True
    def make_lbfgs():return torch.optim.LBFGS(m.parameters(),lr=.5,max_iter=1,max_eval=25,history_size=20,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        nonlocal attempted,completed
        if attempted>=cfg.max_closures:raise Budget('closure_cap')
        attempted+=1
        save_checkpoint()
        if elapsed()>=cfg.train_end:raise Budget('training_time')
        if phase=='adam' and elapsed()>=cfg.adam_end:raise AdamEnd()
        opt.zero_grad();loss=objective()
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise ValueError('nonfinite gradient')
        completed+=1;return loss
    try:
        for outer in range(1,cfg.max_outer+1):
            save_checkpoint()
            if elapsed()>=cfg.train_end:raise Budget('training_time')
            if phase=='adam' and elapsed()>=cfg.adam_end:
                phase='lbfgs';opt=make_lbfgs();unchanged=0
                events.append(dict(event='direct_adam_to_lbfgs',elapsed=elapsed(),unchanged_reset=unchanged))
            before=state(m)
            try:
                if phase=='adam':
                    fraction=min(1.,elapsed()/cfg.adam_end);opt.param_groups[0]['lr']=.0001+.0009*.5*(1+math.cos(math.pi*fraction));closure();opt.step()
                else:opt.step(closure)
                if elapsed()>=cfg.train_end:raise Budget('training_time')
                if phase=='adam' and elapsed()>=cfg.adam_end:raise AdamEnd()
                val=float(objective().detach());post+=1;measured=elapsed()
                if not math.isfinite(val):raise ValueError('nonfinite post-step loss')
                snapshot=state(m);completion=elapsed()
                if completion>=cfg.train_end:raise Budget('training_time')
                if phase=='adam' and completion>=cfg.adam_end:raise AdamEnd()
                accepted+=1
                same=all(torch.equal(v,snapshot[k]) for k,v in before.items())
                unchanged=unchanged+1 if phase=='lbfgs' and same else 0
                history.append(dict(step=outer,phase=phase,loss=val,first_measured_elapsed=measured,completed_elapsed=completion,unchanged_lbfgs_steps=unchanged))
                if val<best:best=val;selected=snapshot;selected_step=outer;selected_measured=measured;selected_completed=completion
                if completion<=cfg.checkpoint and val<checkpoint_best:
                    checkpoint_state=snapshot;checkpoint_best=val;cp_step=outer;cp_measured=measured;cp_completed=completion
                save_checkpoint()
                if phase=='lbfgs' and unchanged>=cfg.unchanged_stop:stop='unchanged_lbfgs_steps';break
            except AdamEnd:
                m.load_state_dict(before);phase='lbfgs';opt=make_lbfgs();unchanged=0
                events.append(dict(event='expired_adam_step_rollback_to_lbfgs',elapsed=elapsed(),unchanged_reset=unchanged))
            except BaseException:m.load_state_dict(before);raise
        else:stop='outer_step_cap'
    except Budget as exc:stop=str(exc)
    except (ValueError,RuntimeError) as exc:failure=type(exc).__name__+': '+str(exc);stop='failed'
    diagnostics={}
    try:
        save_checkpoint(force=True)
        m.load_state_dict(selected);torch.save(selected,out/'selected.pt')
        write(out/'history.json',history);write(out/'events.json',events)
        if elapsed()>=cfg.total:raise Budget('finalization_budget')
        diagnostics=physics_diagnostics(p,m,batch)
        for n,h in sources.items():assert sha(ROOT/n)==h,n
    except (Budget,ValueError,RuntimeError,AssertionError,OSError) as exc:failure=type(exc).__name__+': '+str(exc);stop='finalization_failed'
    result=dict(seed=seed,parameters=12739,stop=stop,failure=failure,eligible=failure is None and accepted>0,selected_training_loss=best,selected_outer_step=selected_step,selected_first_measured_elapsed=selected_measured,selected_completed_elapsed=selected_completed,outer_attempts=outer,accepted_steps=accepted,attempted_closures=attempted,completed_closures=completed,post_step_evaluations=post,diagnostics=diagnostics,checkpoint600_exists=(out/'checkpoint600.pt').exists(),reference_access=False,wall_seconds=elapsed(),cpu_seconds=time.process_time()-cpu,budget_seconds=cfg.total,settings=cfg.__dict__)
    write(out/'fit.json',result)
    if elapsed()>cfg.total:result.update(eligible=False,stop='budget_overrun',failure='Total process budget includes final writes');write(out/'fit.json',result)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--sampling',choices=('dense','sparse'),default='dense');args=parser.parse_args(argv)
    assert args.seed in SEEDS
    if os.path.lexists(args.out):raise FileExistsError('Refusing existing output path: '+str(args.out))
    args.out=args.out.resolve()
    assert args.out!=ROOT and ROOT not in args.out.parents,'Outputs must be outside immutable source tree'
    # Atomic ownership claim is outside the failure handler; no existing output is writable here.
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_interop_threads(1)
        result=_fit_claimed(args.seed,args.out,PROD,sampling=args.sampling)
    except (ValueError,RuntimeError,AssertionError,OSError) as exc:
        result=dict(seed=args.seed,eligible=False,stop='failed',failure=type(exc).__name__+': '+str(exc),reference_access=False,wall_seconds=time.monotonic()-START);write(args.out/'fit.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)
    return 1 if result.get('failure') or not result.get('eligible',False) else 0
if __name__=='__main__':raise SystemExit(main())
