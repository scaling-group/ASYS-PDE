"""Continuous residual and independent mass/moment/flux quadrature."""
import numpy as np
import torch
import problem
h = problem.KSHelpers({}, torch.device("cpu"))

def temporal(model,times,cells):
 rows=[]
 for chunk in times.split(2):
  ut,diff,chem,u,measure=h.continuous_parts(model,cells,chunk)
  fields=(ut-diff-chem,ut,diff,chem,diff+chem)
  integrals=[(f.square()*measure).sum(1).detach() for f in fields]
  plain=[q.sqrt() for q in integrals];floored=[q.clamp_min(1e-30).sqrt() for q in integrals]
  for j,t in enumerate(chunk):
   r,nt,nd,nc,nrhs=(float(q[j]) for q in plain)
   rf,tf,df,cf,_=(float(q[j]) for q in floored)
   rr=h._radial_grid(cells)[0]
   core=float((fields[0][j].detach().square()*measure*(rr<.3)).sum())
   rows.append({'t':float(t),'rho_as_public':rf/(tf+df+cf+1e-10),'residual_radial_L2':r,'ut_radial_L2':nt,'diffusion_radial_L2':nd,'chemistry_radial_L2':nc,'rhs_radial_L2':nrhs,'raw_radial_area_mean_MSE':float(integrals[0][j]/measure.sum()),'residual_squared_fraction_inside_r_point3':core/max(float(integrals[0][j]),1e-300),'u_min':float(u[j].detach().min()),'u_max':float(u[j].detach().max())})
 return rows


def aggregate(rows):
 vals=np.array([r['rho_as_public'] for r in rows]);raw=np.array([r['raw_radial_area_mean_MSE'] for r in rows]);positive=np.array([r['t']>0 for r in rows]);k=int(vals.argmax())
 return {'n_times':len(rows),'rho_geometric_mean':float(np.exp(np.log(vals.clip(1e-12)).mean())),'positive_rho_geometric_mean':float(np.exp(np.log(vals[positive].clip(1e-12)).mean())),'rho_maximum':float(vals[k]),'rho_worst_time':rows[k]['t'],'raw_MSE_temporal_arithmetic_mean':float(raw.mean()),'raw_MSE_maximum':float(raw.max())}


def physical(model,times,cells):
 _,faces,_=h._radial_grid(cells);z,w=np.polynomial.legendre.leggauss(8);z=torch.tensor(z);w=torch.tensor(w);half=(faces[1:]-faces[:-1])/2
 r=(faces[1:]+faces[:-1])[:,None]/2+half[:,None]*z;rw=half[:,None]*w*r
 rows=[]
 for t in times:
  points=torch.stack((r.flatten(),torch.full_like(r.flatten(),float(t))),1).requires_grad_(True)
  value=problem._scalar_field(model(points)).reshape_as(r);ut=problem._grad(value.reshape(-1,1),points)[:,1].reshape_as(r)
  A=float((value*rw).sum().detach());Adot=float((ut*rw).sum().detach());I=float((value*rw*r.square()).sum().detach());Idot=float((ut*rw*r.square()).sum().detach())
  cm=(value*rw).sum(1).cumsum(0);cmt=(ut*rw).sum(1).cumsum(0)
  fp=torch.stack((faces[1:],torch.full_like(faces[1:],float(t))),1).requires_grad_(True);fv=problem._scalar_field(model(fp))[:,0];fg=problem._grad(fv[:,None],fp)[:,0]
  defect=cmt-faces[1:]*fg-fv*cm;k=int(defect.detach().abs().argmax())
  endpoints=torch.tensor([[0.,float(t)],[30.,float(t)]],requires_grad=True);ev=problem._scalar_field(model(endpoints));eg=problem._grad(ev,endpoints);err=problem._grad(eg[:,:1],endpoints)[:,0];ett=problem._grad(eg[:,1:],endpoints)[:,1]
  u0,uR=(float(q) for q in ev.detach().flatten());ur0,urR=(float(q) for q in eg[:,0].detach());flux=30*urR+uR*A;rhs=900*flux-1800*uR+4*A-A*A
  rows.append({'t':float(t),'mass_R30':A,'mass_error_from4point75':A-4.75,'mass_rate':Adot,'boundary_flux':flux,'mass_balance_defect':Adot-flux,'second_moment_R30':I,'second_moment_error_from_whole_plane_law':I-(4.75-3.5625*float(t)),'second_moment_rate':Idot,'finite_R_virial_rhs':rhs,'virial_balance_defect':Idot-rhs,'cumulative_mass_equation_max_defect':float(defect[k].detach().abs()),'cumulative_defect_radius':float(faces[1:][k]),'origin_u':u0,'origin_ur':ur0,'origin_ut':float(eg[0,1].detach()),'origin_urr':float(err[0].detach()),'origin_utt':float(ett[0].detach()),'remote_u':uR,'remote_ur':urR,'origin_pde_defect_if_symmetric':float(eg[0,1].detach())-2*float(err[0].detach())-u0*u0 if ur0==0. else None})
 return rows

