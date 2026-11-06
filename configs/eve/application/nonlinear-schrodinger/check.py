"""Small coordinate/batch checks only; never fit or read a reference."""
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import torch
import yaml
from asys_fit import _build_model, _load_module, _grad


def check_model(model):
    p = torch.tensor([[-1.13, .21, 4.], [.37, .47, 5.], [1.19, .83, 6.]],
                     dtype=torch.float64, requires_grad=True)
    values = model(p)
    single = torch.cat([model(row[None]) for row in p])
    if values.shape != (3, 2) or not torch.isfinite(values).all():
        raise ValueError('Candidate must return three finite u,v rows')
    if not torch.allclose(values, single, rtol=1e-10, atol=1e-11):
        raise ValueError('Candidate depends on unrelated rows in the coordinate batch')
    errors = []
    for column in (0, 1):
        ad = _grad(values[:, column:column+1], p)
        for coordinate in (0, 1):
            shift = torch.zeros_like(p)
            shift[:, coordinate] = 1e-5
            finite = (model(p+shift)[:, column]-model(p-shift)[:, column])/2e-5
            if not torch.allclose(ad[:, coordinate], finite, rtol=2e-4, atol=2e-6):
                raise ValueError('Coordinate gradient disagrees with a finite difference')
            errors.append(float((ad[:, coordinate]-finite).abs().max()))
    return dict(state='passed', parameters=sum(v.numel() for v in model.parameters()),
                buffers=sum(v.numel() for v in model.buffers()), gradient_errors=errors,
                model_calls=0, fits=0, reference_access=False)


def run(root):
    torch.set_num_threads(1)
    model = _build_model(_load_module(root/'model.py', 'checked_model'),
                         yaml.safe_load((root/'config.yaml').read_text()), torch.device('cpu'))
    return check_model(model)


if __name__ == '__main__':
    print(json.dumps(run(Path(__file__).resolve().parent), indent=2))
