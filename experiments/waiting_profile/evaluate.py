"""Assess the frozen conditional logarithmic profile on its stated interval."""
import argparse
import json
from pathlib import Path

import torch

from model import Candidate
import task

ROOT = Path(__file__).resolve().parent


def evaluate():
    model = Candidate()
    model.load_state_dict(torch.load(ROOT/'state_dict.pt', map_location='cpu', weights_only=True), strict=True)
    x = torch.logspace(-5,-2,257,dtype=torch.float64)[:,None]
    f, first, second, residual = task.field_derivatives(model,x)
    scale = (f*second).abs()+first.square()+(task.GAMMA*x*first).abs()+(2*task.GAMMA-1)*f.abs()
    normalized = residual/torch.where(scale>0,scale,torch.ones_like(scale))
    if not torch.isfinite(normalized).all():
        raise ValueError('Non-finite profile defect')
    return dict(normalized_ode_rms=float(normalized.square().mean().sqrt().detach()),
                normalized_ode_max=float(normalized.abs().max().detach()),
                origin_traces=[float(v.detach()[0,0]) for v in task.traces(model)],
                physical_checks=task.physical_checks(model),
                fitted_scalars=sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evaluate(),indent=2,allow_nan=False)+'\n')
