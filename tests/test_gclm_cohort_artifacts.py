"""Original artifact bindings and frozen-profile contracts for the gCLM cohort."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "experiments/gclm"


def test_original_cohort_bytes():
    manifest = json.loads((ROOT / "cohort/source.json").read_text())
    for relative, record in manifest["original_files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == record["sha256"]


def test_nine_frozen_profiles_and_normalization():
    subprocess.run(
        [
            sys.executable,
            "-c",
            r"""
import json
import numpy as np
import torch
from cohort_readout import load, manifest, aggregate

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
original=json.load(open('cohort/expected.json'))['rows']
rows=[]
for declaration, expected in zip(manifest()['rows'],original,strict=True):
    assert declaration['id']==expected['id']
    model=load(declaration)
    before={k:v.clone() for k,v in model.state_dict().items()}
    y=torch.tensor(expected['y'])
    with torch.no_grad():
        values=model(.25,y)
        np.testing.assert_allclose(values.numpy(),expected['profile'],rtol=2e-13,atol=3e-15)
        torch.testing.assert_close(model(.25,-y),-values,rtol=1e-13,atol=1e-15)
        torch.testing.assert_close(model(.25,y[:1]),values[:1],rtol=1e-13,atol=1e-15)
    origin=torch.zeros(1,requires_grad=True)
    assert model(.25,origin).item()==0
    slope=torch.autograd.grad(model(.25,origin).sum(),origin)[0]
    torch.testing.assert_close(slope,torch.ones_like(slope),rtol=2e-13,atol=2e-13)
    assert abs(float(model.beta(.25))-expected['beta'])<2e-15
    assert sum(p.numel() for p in model.parameters())==expected['parameters']
    assert sum(p.numel() for p in model.buffers())==expected['buffers']
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    rows.append(dict(method=declaration['method'],references={
        key:dict(profile_error_percent=100*value['log_grid_relative_l2'])
        for key,value in expected['reference_metrics'].items()}))
summary=aggregate(rows)
for method, count, median in [('ASYS',5,3.8960767841878373e-5),
                              ('SIGS',2,.011917035984025633),
                              ('FEX',2,.008509829982482205)]:
    assert summary[method]['outcomes']==count
    assert abs(summary[method]['references']['fine']['median_profile_error_percent']-median)<1e-14
""",
        ],
        cwd=ROOT,
        check=True,
    )


def test_selected_h3_tail_coefficients():
    pytest.importorskip("mpmath", reason="H3 endpoint algebra uses the baseline extra")
    subprocess.run(
        [
            sys.executable,
            "-c",
            r"""
import json
import math
import torch
from cohort_readout import load, manifest
from evaluate import tail_readout

torch.set_default_dtype(torch.float64)
row=next(row for row in manifest()['rows'] if row['id']=='asys-h3')
model=load(row)
normal=json.load(open('selected/normalization.json'))
expected=json.load(open('selected/expected.json'))
actual=tail_readout(model.raw,normal)
for key in ('C','D','lambda_D','K'):
    assert math.isclose(actual[key],expected[key],rel_tol=1e-13,abs_tol=1e-13), key
assert abs(actual['tail_balance_error_percent']-expected['tail_balance_error_percent'])<1e-10
assert round(actual['tail_balance_error_percent'],4)==.0025
assert actual['endpoint_algebra_forward_max_abs']<1e-12
""",
        ],
        cwd=ROOT,
        check=True,
    )
