"""Numerical qualification; exact equation fixture and negative controls."""
import argparse, json, os, sys
from pathlib import Path
import torch


def qualify(application):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    sys.path.insert(0, str(application))
    import task
    from check import check_field
    class Exact(torch.nn.Module):
        def forward(self, x): return x.square()/6
    class Detached(torch.nn.Module):
        def forward(self, x): return x.detach().square()/6
    class Coupled(torch.nn.Module):
        def forward(self, x): return x.square()/6+x.mean()*1e-3
    x = torch.tensor([[.0013],[.0057],[.0091]])
    error = float(task.field_derivatives(Exact(),x)[3].abs().max())
    assert error < 1e-18, error
    check_field(Exact())
    assert not task.physical_checks(Exact())['passes'], 'Zero cubic must fail the supplied normalization'
    rejected = []
    for label, model in [('detached', Detached()), ('batch_coupled', Coupled())]:
        try: check_field(model)
        except (ValueError, RuntimeError): rejected.append(label)
    assert rejected == ['detached','batch_coupled']
    model = task.load_model(application/'model.py')
    physical, derivatives = task.physical_checks(model), check_field(model)
    assert physical['passes'] and derivatives['passes']
    return {'passes': True, 'exact_quadratic_equation_fixture_error':error,
        'zero_cubic_negative_control_rejected':True, 'negative_controls_rejected':rejected,
        'initial_candidate_physical':physical, 'initial_candidate_derivatives':derivatives}

if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--application',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(qualify(a.application.resolve()),indent=2,allow_nan=False)+'\n')
