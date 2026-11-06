"""Existing FEX controller/pool loop with an explicitly adapted common fitter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(HERE/'kernel')]
import torch
import yaml
import core_fitting
import common_single
from frozen_profile import FrozenProfile, origin_slope
from model import Model
from run import Numerical, run_case, write
from search import Controller, Pool, formula, reward


class CommonNumerical(Numerical):
    def __init__(self, settings):
        self.torch,self.Model,self.Controller,self.Pool = torch,Model,Controller,Pool
        self.formula,self.reward,self.settings = formula,reward,settings
        self.policy=json.loads((HERE/'policy.json').read_text())
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)

    def fit(self,path,config,stages,seconds,initial=None):
        path.mkdir(parents=True,exist_ok=False)
        write(path/'model_config.json',config)
        (path/'model.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        raw=self.Model(config)
        if initial is not None:
            raw.load_state_dict(torch.load(initial/'state_dict.pt',weights_only=True,map_location='cpu'),strict=True)
        phase='terminal' if initial is not None else 'coarse'
        if seconds < self.policy[phase]['seconds']:
            result=dict(completed=False,selection_loss=None,seconds=0.,stop='insufficient_complete_fit_reservation')
            write(path/'fit.json',result)
            return result
        started=time.monotonic()
        _,fit=core_fitting.fit(raw,self.policy,phase)
        eligible=fit['eligible']
        result=dict(completed=eligible,selection_loss=1e6*fit['validation']['fine'] if eligible else None,
                    seconds=time.monotonic()-started,stop=fit['stop'],failure=fit['failure'],fit=fit,
                    reward_loss_scaling=1000000,initialization='same-run-coarse' if initial else 'cold')
        torch.save(raw.state_dict(),path/'state_dict.pt')
        write(path/'formula.json',self.formula(raw))
        if eligible and phase=='terminal':
            amp=fit['validation']['mass'];slope=origin_slope(raw)
            frozen=FrozenProfile(raw,amp,slope)
            torch.save(frozen.state_dict(),path/'frozen_state_dict.pt')
            write(path/'normalization.json',dict(amplitude=amp,origin_slope=slope,coordinate_scale=amp/slope,beta=float(raw.beta())))
        write(path/'fit.json',result)
        return result

    def score(self,path):
        saved=path/'public.json'
        if saved.exists():return json.loads(saved.read_text())
        raw=self.Model(json.loads((path/'model_config.json').read_text()))
        raw.load_state_dict(torch.load(path/'state_dict.pt',weights_only=True,map_location='cpu'),strict=True)
        fit=json.loads((path/'fit.json').read_text())
        n=path/'normalization.json'
        if n.exists():
            normal=json.loads(n.read_text());frozen=FrozenProfile(raw,normal['amplitude'],normal['origin_slope'])
            frozen.load_state_dict(torch.load(path/'frozen_state_dict.pt',weights_only=True,map_location='cpu'),strict=True)
        else:
            frozen=FrozenProfile(raw,fit['fit']['validation']['mass'],origin_slope(raw))
        cfg=json.loads((HERE/'assessment_config.json').read_text())
        assessment=common_single.assess_family(frozen,anchors=(.25,),config=cfg,budget=core_fitting.Work(120,cfg['max_field_elements']))
        good=assessment['status']=='qualified' and assessment.get('score') is not None
        result=dict(admissible=good,score=-assessment['score'] if good else -1e9,assessment=assessment,reference_access=False)
        write(saved,result)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    settings=yaml.safe_load((HERE/'study.yaml').read_text())
    engine=CommonNumerical(settings)
    result=run_case(HERE,2,args.seed,settings,engine,output=args.output)
    path=args.output
    chosen=result['public_selected']
    record=dict(method='exponent-compactified FEX-derived adaptation',selected=chosen,reference_access=False,
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    if chosen:
        selected=path/'terminal'/chosen
        record['frozen_files']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected.iterdir() if p.is_file()}
    write(path/'FROZEN_SELECTION.json',record)
