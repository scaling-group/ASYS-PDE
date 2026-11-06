"""Independent geometric mean-curvature evolution, not the harmonic prediction."""
from pathlib import Path
import numpy as np
import json,csv,time,os
from numerics import R0
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output
out.mkdir(parents=True,exist_ok=False)
start=time.monotonic();cpu=time.process_time()
def run(a,n,h):
 theta=2*np.pi*np.arange(n)/n;k=np.fft.fftfreq(n,d=1/n);radius=(1+a*np.cos(2*theta))/np.sqrt(1+a*a/2);radii=[];rows=[];tau=np.linspace(0,.15,61)
 def rhs(r):
  f=np.fft.fft(r);rt=np.fft.ifft(1j*k*f).real;rtt=np.fft.ifft(-k*k*f).real
  return -(r*r+2*rt*rt-r*rtt)/(r*(r*r+rt*rt))
 for i,z in enumerate(tau):
  if i:
   count=int(np.ceil((z-tau[i-1])/h-1e-10));dt=(z-tau[i-1])/count
   for _ in range(count):
    v1=rhs(radius);v2=rhs(radius+dt*v1/2);v3=rhs(radius+dt*v2/2);v4=rhs(radius+dt*v3);radius+=dt*(v1+2*v2+2*v3+v4)/6
  assert np.isfinite(radius).all() and radius.min()>0
  r=R0*radius;r0=r.mean();e2=2*np.mean(r*np.cos(2*theta))/r0;e4=2*np.mean(r*np.cos(4*theta))/r0;s=1-2*z
  rows.append(dict(index=i,tau=float(z),s=float(s),r0=float(r0),e2=float(e2),e4=float(e4),R_area=float(np.sqrt(np.mean(r*r))),area=float(np.pi*np.mean(r*r)),predicted_e4=float(11/20*a*a*(s*s-s**7)),area_law_error=float(np.mean(radius**2)-s)))
  radii.append(r)
 name=f'a{a:.2f}-n{n}'
 with (out/(name+'.csv')).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(out/(name+'.npz'),theta=theta,r=radii,tau=tau)
 return dict(name=name,n=n,dt_tau=h,amplitude=a,max_area_law_error=max(abs(x['area_law_error']) for x in rows),endpoint=rows[-1])
records=[run(a,128,1e-4) for a in [0.,.04,.08]]+[run(.08,256,2.5e-5)]
assert records[0]['max_area_law_error']<1e-10
(out/'RESULT.json').write_text(json.dumps(dict(status='complete',method='polar mean curvature flow r_tau=-(r^2+2r_theta^2-r*r_thetatheta)/(r*(r^2+r_theta^2)), dimensionless radius r/R0 and tau=eps^2*t/R0^2; FFT derivatives and explicit RK4',records=records,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu),indent=2)+'\n')
