"""Assess all three saved single-session outputs for a selected task."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from frozen_profile import FrozenProfile

ROOT=Path(__file__).resolve().parent


def evaluate(task, reference):
    if task=='gclm':
        sys.path.insert(0,str(ROOT.parents[1]/'scripts/references'))
        from fixed_point_reference import CanonicalProfile, LogProfile
        with np.load(reference/'fine/profile.npz',allow_pickle=False) as data:
            raw=LogProfile(data['w'],data['log_values'],float(data['tail_power']))
            y=np.logspace(-3,3,257)
            truth=CanonicalProfile(raw,float(data['b'])).evaluate(y)[0]
    else:
        sys.path.insert(0,str(ROOT.parent/'fujita'))
        from evaluate import assess
        with np.load(reference,allow_pickle=False) as data:
            radius,times,truth=data['r'],data['t'],data['u']
    rows=[]
    for attempt in (1,2,3):
        name=f'{task}-{attempt}';folder=ROOT/name
        spec=importlib.util.spec_from_file_location(name,folder/'model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if task=='gclm':
            config=json.loads((folder/'model_config.json').read_text())
            normalization=json.loads((folder/'normalization.json').read_text())
            model=FrozenProfile(module.build_model(config),normalization['amplitude'],normalization['origin_slope'])
        else:
            model=module.Candidate(yaml.safe_load((folder/'config.yaml').read_text())).double()
        model.load_state_dict(torch.load(folder/'state_dict.pt',map_location='cpu',weights_only=True),strict=True)
        model.eval()
        if task=='gclm':
            with torch.no_grad():values=model(.25,torch.tensor(y)).numpy()
            if not np.isfinite(values).all():raise ValueError('Non-finite frozen profile')
            result=dict(profile_error_percent=100*float(np.linalg.norm(values-truth)/np.linalg.norm(truth)))
        else:
            result=assess(model,radius,times,truth)
        rows.append(dict(candidate=name,**result))
    return dict(units='Relative errors are percentages.',rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('gclm','fujita'),required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    result=evaluate(args.task,args.reference)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
