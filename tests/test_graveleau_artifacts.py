"""Authenticity and scientific contracts of the recovered pressure construction."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments/graveleau"


def test_original_artifact_bytes():
    manifest = json.loads((ROOT / "source.json").read_text())
    for relative, item in manifest["original_files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == item["sha256"]


def test_frozen_field_readout_and_profile_component():
    subprocess.run(
        [
            sys.executable,
            "-c",
            r"""
import json
import math
import torch
from evaluate import evaluate, load
from diagnostics import END
from profile import fitted_coefficients, profile
from checks import initial_checks, direct_jet

expected=json.load(open('expected.json'))['field_readout']
actual=evaluate()['field_readout']
assert actual['n_parameters']==24
for key in ('ic_max_absolute_error', 'true_initial_pde_tangent_max_absolute_error',
            'sampled_mass_range_fraction_of_initial'):
    assert math.isclose(actual[key],expected[key],rel_tol=2e-10,abs_tol=1e-13),key
for measured,original in zip(actual['rows'],expected['rows'],strict=True):
    for key in original:
        assert math.isclose(measured[key],original[key],rel_tol=2e-10,abs_tol=1e-13),key
model=load()
for key,value in initial_checks(model).items():
    assert value < 2e-7,(key,value)
for fraction in (.0,.1,.5,1.):
    time=torch.tensor([[fraction*END]],requires_grad=True)
    radius,speed,beta=model.front(time)
    actual_speed=-torch.autograd.grad(radius.sum(),time)[0]
    torch.testing.assert_close(actual_speed,speed,rtol=1e-13,atol=1e-13)
    edge=torch.cat((radius.detach(),time.detach()),1)
    torch.testing.assert_close(direct_jet(model,edge)[1],speed,rtol=1e-12,atol=1e-13)
    outer=torch.tensor([[6.,fraction*END]])
    assert abs(float(direct_jet(model,outer)[1].detach())) < 1e-13
    distance=torch.tensor([[.001,.01,.1,1.]])
    full=model.shell_exponent(distance,time,radius,speed,beta)
    shell=model.shell_exponent(distance,time,radius,torch.zeros_like(speed),beta)
    b,coefficients=fitted_coefficients(model,fraction*END)
    component=speed*radius/1.6*profile(1+distance/radius,b,coefficients)
    torch.testing.assert_close(full-shell,component,rtol=1e-11,atol=1e-13)
""",
        ],
        cwd=ROOT,
        check=True,
    )
