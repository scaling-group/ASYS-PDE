"""Numerical contracts for the portable manuscript artifacts."""

import subprocess
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def run_study(name, code):
    subprocess.run([sys.executable, "-c", code], cwd=EXPERIMENTS / name, check=True)


def test_ac_time_pool_and_periodic_contours():
    run_study(
        "allen_cahn",
        """
import numpy as np
from metrics import weights, interface, pool
assert np.array_equal(weights([0,1,3]), [.5,1.5,1.])
d=interface(np.array([[.999,0]]), np.array([[-.999,0]]))
assert abs(d['interface_rms']-.002)<1e-14
rows=[dict(weight=1.,E=1.,R=1.,band_E=1.,band_R=1.,Linf=1.,interface_rms=None,interface_max=None),
      dict(weight=3.,E=0.,R=1.,band_E=0.,band_R=1.,Linf=0.,interface_rms=None,interface_max=None)]
assert pool(rows)['field_error_percent']==50.
""",
    )


def test_fujita_clipped_energy_quadrature():
    run_study(
        "fujita",
        """
import numpy as np
from metrics import weights
x=np.array([0.,.3,1.,5.]); w=weights(x,.5)
assert abs(w.sum()-.5)<1e-14
assert abs(w@x-.125)<1e-14
assert np.array_equal(weights([0.,1.,3.]), [.5,1.5,1.])
""",
    )


def test_fujita_sampling_and_paired_initialization():
    run_study(
        "fujita/neural",
        """
import sys
import torch
torch.set_default_dtype(torch.float64)
sys.path.insert(0,'sources')
from fujita_problem import Problem
from neural import Neural
from dense_sampling import dense_batch, verify_initial
p=Problem('cpu'); a,ac,am=dense_batch(p,1); b,bc,bm=dense_batch(p,8)
assert a[0].shape==(512,2) and ac.shape==(208,2)
assert b[0].shape==(4096,2) and bc.shape==(1664,2)
assert torch.equal(a[0],b[0][:512]) and torch.equal(ac,bc[:208])
assert am['origin_count']==16 and bm['origin_count']==128
for seed in (2026110501,2026110511,2026110521): verify_initial(Neural(seed),seed)
""",
    )


def test_graveleau_front_and_constant_defect():
    run_study(
        "graveleau",
        """
import torch
from profile import profile
torch.set_default_dtype(torch.float64)
beta=.75; p=2-1/beta
for repaired in (False,True):
 x=torch.tensor([1.,1e5],requires_grad=True)
 y=profile(x,beta,[0.],repaired)
 first=torch.autograd.grad(y.sum(),x,create_graph=True)[0]
 second=torch.autograd.grad(first.sum(),x)[0]
 residual=y*(second+first/x)+first**2-x*first+p*y
 assert y[0]==0 and abs(float(first[0])-1)<1e-14
 if repaired: assert abs(float(residual[-1]))<.001
 else: assert abs(float(residual[-1])+1)<.001
""",
    )


def test_waiting_profile_frozen_traces():
    run_study(
        "waiting_profile",
        """
import torch
torch.set_default_dtype(torch.float64)
from evaluate import evaluate
r=evaluate()
assert r['fitted_scalars']==18 and r['physical_checks']['passes']
assert r['normalized_ode_rms']<3e-16
assert all(abs(a-b)<1e-12 for a,b in zip(r['origin_traces'],[0.,0.,1/3,6.]))
""",
    )


def test_stationary_printed_coefficients_and_origin_curvature():
    run_study(
        "fujita_stationary",
        """
import torch
torch.set_default_dtype(torch.float64)
from evaluate import load, evaluate
m=load(); x=torch.zeros((1,1),requires_grad=True); y=m(x)
first=torch.autograd.grad(y.sum(),x,create_graph=True)[0]
second=torch.autograd.grad(first.sum(),x)[0]
assert abs(float(y)-6.63471855955)<5e-12
assert float(first)==0
assert abs(float(second)+57.7479140847)<5e-10
assert sum(p.numel() for p in m.parameters())==38
def reject_candidate_residual(*args):
    raise AssertionError('candidate residual used')
type(m).equation_terms=reject_candidate_residual
r=evaluate()
assert r['peak_sampled_ode_defect']<6e-12
assert abs(r['endpoint_robin_defect']+4.35304203612e-8)<1e-15
assert r['diagnostic_grid']['points']==11001
assert r['origin_compatibility_absolute']<3e-14
assert r['batch_max_absolute_difference']==0
assert r['mode_max_absolute_difference']==0
""",
    )


def test_ks_mass_reference_jacobian():
    run_study(
        "keller_segel",
        """
from reference import check_jacobian
assert check_jacobian()<1e-10
""",
    )


def test_nls_pooled_spatial_mean_energies():
    run_study(
        "nonlinear_schrodinger",
        """
from metrics import pool
rows=[dict(width=3.5,squared_error=1.,reference_energy=1.,reference_correction_energy=2.),
      dict(width=7.,squared_error=0.,reference_energy=8.,reference_correction_energy=2.)]
r=pool(rows)
assert abs(r['field_relative']-1/3)<1e-15
assert r['correction_relative']==.5
""",
    )


def test_ks_continuation_initialization_and_original_probes():
    run_study(
        "keller_segel",
        """
import importlib.util
import numpy as np
import torch
import yaml
from evaluate import ROOT, load
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
folder=ROOT/'selected/gaussian_continued'
spec=importlib.util.spec_from_file_location('ks_initializer',folder/'model.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
initial=source.Candidate(yaml.safe_load((folder/'config.yaml').read_text()))
original=load('gaussian')
assert all(torch.equal(v,original.state_dict()[k]) for k,v in initial.state_dict().items())
for name in ('gaussian_continued','laguerre48'):
    model=load(name)
    with np.load(ROOT/'selected'/name/'probe.npz',allow_pickle=False) as saved:
        points=torch.tensor(saved['points'],requires_grad=True)
        value=model(points)[:,0]
        first=torch.autograd.grad(value.sum(),points,create_graph=True)[0]
        drr=torch.autograd.grad(first[:,0].sum(),points)[0][:,0]
        for key,result in (('value',value),('dr',first[:,0]),('dt',first[:,1]),('drr',drr)):
            np.testing.assert_allclose(result.detach().numpy().ravel(),saved[key].ravel(),rtol=1e-10,atol=1e-12)
""",
    )
