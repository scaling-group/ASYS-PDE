"""Refined mass-drift comparison for prescribed P and P+E amplitudes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import Candidate
from metrics import predict

ROOT = Path(__file__).resolve().parent


def evaluate(widths, time_counts):
    config=json.loads((ROOT/'config.json').read_text())
    rows=[]
    for order in (1,2):
        model=Candidate(dict(config,fixed_predictor_order=order)).eval()
        for width in widths:
            nx=4096 if width<=4.5 else 2048
            x=-width+2*width*np.arange(nx)/nx
            for nt in time_counts:
                times=np.linspace(0,np.pi/2,nt)
                field=predict(model,x,times,width)
                mass=2*width*np.mean(abs(field)**2,axis=1)
                target=8*np.tanh(width)
                rows.append(dict(order=order,width=width,nx=nx,nt=nt,
                    mass_initial=float(mass[0]),
                    mass_drift_relative=float(np.max(abs(mass-mass[0]))/mass[0]),
                    mass_target_relative=float(np.max(abs(mass-target))/target)))
    return dict(units='Relative mass errors are fractions; multiply by 1e6 for ppm.',rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--widths',type=float,nargs='+',choices=(3.5,4.,4.5,5.,6.,7.),default=(3.5,4.,4.5,5.,6.,7.))
    parser.add_argument('--time-counts',type=int,nargs='+',choices=(321,641,1281),default=(321,))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    result=evaluate(args.widths,args.time_counts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
