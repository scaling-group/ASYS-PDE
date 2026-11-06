from pathlib import Path
import argparse,json,time,os,hashlib,csv,resource
import numpy as np
from numerics import R0,ETDRK4,signed_distance,contour
ROOT=Path(__file__).resolve().parents[1]
def guard():
 for n,h in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--a',type=float,required=True);p.add_argument('--eps',type=float,required=True);p.add_argument('--n',type=int,required=True);p.add_argument('--dt',type=float,required=True);args=p.parse_args();guard()
out=args.output;out.mkdir(parents=True,exist_ok=False);wall=time.monotonic();cpu=time.process_time()
settings=dict(**{k:v for k,v in vars(args).items() if k!='output'},R0=R0,equation='u_t=eps^2 Laplacian(u)+u-u^3',domain='[-1,1)^2 periodic',initial='tanh(actual signed Euclidean distance/(sqrt(2)*eps)), positive inside',source_manifest_sha256=hashlib.sha256((ROOT/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),status='running')
(out/'SETTINGS.json').write_text(json.dumps(settings,indent=2)+'\n')
distance,meta=signed_distance(args.n,args.a);u=np.tanh(distance/(np.sqrt(2)*args.eps));settings['distance']=meta;settings['initial_array_sha256']=hashlib.sha256(u.tobytes()).hexdigest()
if args.case=='eps02-a08':
 d2,_=signed_distance(args.n,args.a,samples=32768);settings['distance_seed_refinement_Linf_near_interface']=float(np.max(abs(d2-distance)[abs(distance)<.12]))
solver=ETDRK4(u,args.eps);svalues=np.linspace(1,.7,61);times=(1-svalues)*R0**2/(2*args.eps**2)
rows=[];curves=[];fields=[];field_indices=[];extract_checks=[];steps=0
for i,(target,s) in enumerate(zip(times,svalues)):
 if i:
  interval=target-times[i-1];count=int(np.ceil(interval/args.dt-1e-10));dt=interval/count;solver.advance(count,dt);steps+=count
 u=solver.field();assert np.isfinite(u).all()
 radii,row=contour(u);curves.append(radii)
 # Locked prediction is readout-only, never supplied to the PDE integrator.
 pred=11/20*args.a**2*(s**2-s**7);sa=(row['R_area']/R0)**2;pred_area=11/20*args.a**2*(sa**2-sa**7)
 row.update(index=i,t=float(target),s_prescribed=float(s),s_measured_area=float(sa),predicted_e4=float(pred),predicted_e4_measured_area=float(pred_area),e4_error=float(row['e4']-pred),e4_absolute_error=float(abs(row['e4']-pred)),e4_measured_area_absolute_error=float(abs(row['e4']-pred_area)),e4_over_a2=float(row['e4']/args.a**2) if args.a else None,u_min=float(u.min()),u_max=float(u.max()),steps=steps)
 rows.append(row)
 if i%10==0:
  fields.append(u.copy());field_indices.append(i)
  if args.case=='eps02-a08':
   _,fine=contour(u,4096,5);_,cubic=contour(u,2048,3)
   extract_checks.append(dict(index=i,angular2048_to4096_e4_difference=float(fine['e4']-row['e4']),quintic_minus_cubic_e4=float(row['e4']-cubic['e4'])))
  print(json.dumps(dict(case=args.case,index=i,t=float(target),steps=steps,seconds=time.monotonic()-wall)),flush=True)
with (out/'metrics.csv').open('x',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savez_compressed(out/'contours.npz',theta=2*np.pi*np.arange(2048)/2048,r=np.stack(curves),times=times,s=svalues)
np.savez_compressed(out/'fields.npz',axis=-1+2*np.arange(args.n)/args.n,u=np.stack(fields),indices=field_indices,times=times[field_indices])
guard();settings.update(status='complete',steps=steps,saved_times=61,wall_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,extraction_checks=extract_checks,arrays={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in out.iterdir() if p.suffix in ['.csv','.npz']})
(out/'RESULT.json').write_text(json.dumps(settings,indent=2,allow_nan=False)+'\n')
