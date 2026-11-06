"""High-precision audit of the formal construction, separate from float64 fit.

Use the exact rational transport before its storage rounding to expose the
nonzero remainder hidden by ordinary precision. This is an audit of the ansatz,
not a numerical reference solution of the ODE. No marching or shooting occurs.
"""
import runpy, json, sys
from pathlib import Path
import mpmath as mp
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
ns=runpy.run_path(str(ROOT/'polyhomogeneous.py'))
q,z,c,s=ns['q'],ns['z'],ns['c'],ns['s']
import torch
params=torch.load(ROOT.parent/'state_dict.pt',map_location='cpu',weights_only=True)['envelope'].tolist()
mp.mp.dps=180
cm=mp.mpf(str(params[0]))
c_exact=s.Rational(str(params[0]))
a=[mp.mpf(str(s.N(q[k].subs({z:0,c:c_exact}),185))) for k in range(1,12)]
b=[mp.mpf(str(s.N(q[k].coeff(z,1),185))) for k in range(1,12)]
hcoef=[mp.mpf(str(v)) for v in params[1:]]

def quantities(u):
    zz=2*u-1
    tt=[mp.mpf(1),zz]; dt=[mp.mpf(0),mp.mpf(2)]; ddt=[mp.mpf(0),mp.mpf(0)]
    for k in range(2,len(hcoef)):
        tt.append(2*zz*tt[-1]-tt[-2])
        dt.append(4*tt[-2]+2*zz*dt[-1]-dt[-2])
        ddt.append(8*dt[-2]+2*zz*ddt[-1]-ddt[-2])
    h=sum(v*t for v,t in zip(hcoef,tt))
    dh=sum(v*t for v,t in zip(hcoef,dt))
    ddh=sum(v*t for v,t in zip(hcoef,ddt))
    qv=u**12*h
    dq=u**12*(12*h+u*dh)
    ddq=u**12*(144*h+25*u*dh+u*u*ddh)
    for k,(ak,bk) in enumerate(zip(a,b),1):
        p=ak+bk*mp.log(u)
        qv+=u**k*p
        dq+=u**k*(k*p+bk)
        ddq+=u**k*(k*k*p+2*k*bk)
    y=mp.exp(qv)/6
    A=2+dq; D=2+3*dq+ddq+dq*dq
    R=y*(D+A*A)-mp.mpf(7)/3*A+mp.mpf(11)/3
    S=abs(y*D)+y*A*A+mp.mpf(7)/3*abs(A)+mp.mpf(11)/3
    return R/S,qv

rows=[]
for us in ['1','.1','.01','.001','.0001','.000001','.00000001']:
    u=mp.mpf(us)
    r,qv=quantities(u)
    rows.append({'u':us,'xi':mp.nstr(u/100,12),
        'ideal_transport_normalized_residual':mp.nstr(r,24),
        'residual_over_u12_log_squared':mp.nstr(r/(u**12*mp.log(u)**2),24) if u!=1 else None})
expected=s.expand(ns['next_forcing']).coeff(z,2)/56
report={'precision_digits':mp.mp.dps,'coefficient_scope':'exact rational transport; learned smooth remainder as decimal constants',
    'probe_results':rows,
    'asymptotic_residual_ratio_limit':str(s.N(expected,30)),
    'eighth_derivative_log_coefficient':-1241866080*40320,
    'interpretation':'The high-precision residual is nonzero; the probes approach the logarithmic ratio whose limit is independently derived exactly. Float64 residuals at roundoff cannot certify an exact solution. The true eighth derivative diverges like 8!*b*log(xi/.01).'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
