"""Official public fitness from frozen source/configuration and registered state.

Run in a fresh interpreter. Fitting losses and optimizers are never imported.
Canonical operators are loaded again after candidate construction/state loading.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import torch
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def reconstruct(solver, state):
    sys.path.insert(0,str(solver))
    constructor=module(solver/'task.py','task')
    cfg=yaml.safe_load((solver/'config.yaml').read_text())
    torch.manual_seed(int(cfg.get('seed',0)))
    model=constructor.load_model(solver/'model.py')
    model.load_state_dict(torch.load(state,map_location='cpu',weights_only=True))
    # A constructor or custom state loader may have changed the imported helpers.
    # Restore aliases held by the model, then use a separate canonical namespace.
    task=module(solver/'task.py','_graveleau_canonical_task')
    constructor.__dict__.update(task.__dict__)
    sys.modules['task']=task
    checker=module(solver/'check.py','_graveleau_canonical_check')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    return model,task,checker


def evaluate(solver, state, eligibility_only=False):
    inputs={n:digest(solver/n) for n in
            ('model.py','loss.py','optim.py','config.yaml','task.py','check.py')}
    state_hash=digest(state)
    model,task,checker=reconstruct(solver,state)
    physical=task.physical_checks(model)
    derivatives=checker.check_field(model)
    result={'physical':physical,'derivatives':derivatives,
            'input_sha256':inputs,'state_sha256':state_hash,
            'scorer_sha256':digest(Path(__file__).resolve()),
            'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'buffer_scalars':sum(b.numel() for b in model.buffers())}
    if not eligibility_only:
        terms={k:float(v.detach()) for k,v in task.terms(model,task.samples(991,2)).items()}
        penalty=0 if physical['passes'] and derivatives['passes'] else 1000
        corrected={'physics':-penalty-terms['bulk_physics']-terms['early_physics']-terms['weak'],
                   'ic':-penalty-terms['initial'],'ic_tangent':-penalty-terms['tangent'],
                   'bc':-penalty-terms['boundary']}
        result.update(public_terms=terms,corrected_scores=corrected)
    if inputs!={n:digest(solver/n) for n in inputs} or state_hash!=digest(state):
        raise ValueError('Frozen scoring inputs changed during reconstruction/evaluation')
    return result


def score_fit(fit):
    start=time.monotonic()
    metadata_path=fit/'fit.json'
    metadata_hash=digest(metadata_path)
    metadata=json.loads(metadata_path.read_text())
    for name,expected in metadata['input_sha256'].items():
        if digest(fit/'frozen_candidate'/name)!=expected:
            raise ValueError('Frozen source/configuration/operator digest mismatch: '+name)
    for name,expected in metadata['protected_evaluator_sha256'].items():
        if digest(fit/'frozen_evaluator'/name)!=expected:
            raise ValueError('Frozen evaluator digest mismatch: '+name)
    if digest(Path(__file__).resolve())!=metadata['protected_evaluator_sha256']['score.py']:
        raise ValueError('Use the exact frozen official scorer')
    for name,expected in metadata['state_sha256'].items():
        if digest(fit/name)!=expected:
            raise ValueError('Frozen state digest mismatch: '+name)
    initial=evaluate(fit/'frozen_candidate',fit/'initial_state.pt',eligibility_only=True)
    selected=evaluate(fit/'frozen_candidate',fit/'selected_state.pt')
    vector=selected[metadata['public_judge']+'_scores']
    report=dict(metadata,initial_checks=initial['physical'],initial_derivatives=initial['derivatives'],
                selected_checks=selected['physical'],selected_derivatives=selected['derivatives'],
                public_terms=selected['public_terms'],public_scores=vector,score=sum(vector.values())/4,
                corrected_scores=selected['corrected_scores'],
                parameters=selected['parameters'],buffer_scalars=selected['buffer_scalars'],
                fitting_record_sha256=metadata_hash,scoring_elapsed_seconds=time.monotonic()-start,
                scoring_contract='Fresh frozen source/configuration/registered-state reconstruction; canonical protected operators; fitting loss/optimizer not imported')
    report['evaluation_elapsed_seconds']=metadata['fitting_evaluation_elapsed_seconds']+report['scoring_elapsed_seconds']
    if metadata_hash!=digest(metadata_path):
        raise ValueError('Fitting record changed during scoring')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fit',type=Path)
    parser.add_argument('--solver',type=Path)
    parser.add_argument('--state',type=Path)
    parser.add_argument('--eligibility-only',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--score',type=Path)
    args=parser.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    if args.fit:
        result=score_fit(args.fit.resolve())
    elif args.solver and args.state:
        result=evaluate(args.solver.resolve(),args.state.resolve(),args.eligibility_only)
    else:
        parser.error('Provide --fit or both --solver and --state')
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output: args.output.write_text(encoded)
    if args.score:
        score=dict(result['public_scores'],score=result['score'],
                   physical_checks_pass=result['selected_checks']['passes'],
                   summary='Frozen conditional waiting-profile equation and supplied normalization')
        args.score.write_text(json.dumps(score,allow_nan=False)+'\n')
    print(encoded,end='')
