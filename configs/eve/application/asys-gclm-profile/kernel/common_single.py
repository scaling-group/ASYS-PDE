"""Independent fixed-state gCLM assessment; no fitting, references or re-gauging.

Positive-axis log quadrature is independent of training's reciprocal quadrature.
'qualified' means this finite numerical instrument resolved its measurements,
not that the candidate solves the PDE or approximates a protected reference.
"""
import math
import time
from functools import lru_cache
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np
import torch

VERSION = "common-assessment-v1r5"
PUBLIC_ANCHORS = (0.25,)

@dataclass(frozen=True)
class Config:
    extents: tuple = (18., 36., 72., 144., 288., 576.)
    survey_step: float = 0.25
    survey_refinements: int = 6
    mass_tolerance: float = 1e-8
    gauge_tolerance: float = 1e-6
    quantile_lower: float = 0.001
    quantile_upper: float = 0.999
    guard: float = 2.
    coverage_sites: int = 257
    transition_sites: int = 64
    operator_panel_width: float = 0.5
    operator_orders: tuple = (16, 32, 64)
    operator_extension: float = 8.
    operator_tolerance: float = 1e-7
    score_tolerance: float = 1e-5
    fixed_sites: int = 33
    batch_sites: int = 16
    field_batch: int = 65536
    max_field_elements: int = 100000000
    max_operator_sites: int = 50000

class Unresolved(Exception):
    pass

class Rejected(Exception):
    pass

def _config(config):
    if isinstance(config, Config):
        cfg = config
    else:
        cfg = Config(**dict(config or {}))
    if (cfg.coverage_sites < 17 or cfg.survey_refinements < 1
        or cfg.batch_sites < 1 or cfg.field_batch < 1
        or cfg.operator_panel_width <= 0 or cfg.survey_step <= 0
        or cfg.operator_extension <= 0 or cfg.transition_sites < 0
        or len(cfg.extents) < 2 or any(not 0 < e <= 576 for e in cfg.extents)
        or any(b <= a for a,b in zip(cfg.extents,cfg.extents[1:]))
        or len(cfg.operator_orders) < 3
        or any(n < 4 for n in cfg.operator_orders)
        or any(b <= a for a,b in zip(cfg.operator_orders,cfg.operator_orders[1:]))
        or not 0 < cfg.quantile_lower < cfg.quantile_upper < 1
        or min(cfg.mass_tolerance,cfg.gauge_tolerance,cfg.operator_tolerance,cfg.score_tolerance) <= 0):
        raise ValueError("Invalid fixed evaluator configuration")
    return cfg

class Context:
    def __init__(self, family, anchor, cfg, budget, counters):
        self.family, self.anchor, self.cfg = family, anchor, cfg
        self.budget, self.counters = budget, counters
        self.details = {}

    def check(self):
        if self.budget is not None:
            self.budget.check()
        if self.counters["field_elements"] > self.cfg.max_field_elements:
            raise Unresolved("field evaluation limit")
        if self.counters["operator_sites"] > self.cfg.max_operator_sites:
            raise Unresolved("operator site limit")

    def field(self, t, derivative=False, signed=False):
        t = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(t)) or np.max(np.abs(t), initial=0) > 700:
            raise Unresolved("unrepresentable requested log radius")
        shape = t.shape
        result, derivatives = [], []
        for start in range(0,t.size,self.cfg.field_batch):
            self.check()
            y = torch.tensor(np.exp(t.ravel()[start:start+self.cfg.field_batch]),
                             dtype=torch.float64, requires_grad=derivative)
            self.counters["field_calls"] += 1
            self.counters["field_elements"] += y.numel()
            with torch.enable_grad() if derivative else torch.no_grad():
                f = self.family(self.anchor, -y if signed else y)
                if not isinstance(f,torch.Tensor) or f.dtype != torch.float64 or f.device.type != "cpu" or f.shape != y.shape:
                    raise Rejected("forward must preserve CPU float64 input shape")
                if derivative:
                    if not f.requires_grad:
                        raise Rejected("profile has no differentiable radius dependence")
                    fp = torch.autograd.grad(f.sum(), y, create_graph=False)[0]
                    D = y*fp
                    self.counters["derivative_calls"] += 1
                    derivatives.append(D.detach().numpy())
            result.append(f.detach().numpy())
        f = np.concatenate(result).reshape(shape)
        if not np.all(np.isfinite(f)):
            raise Unresolved("nonfinite profile values")
        if not signed and np.any(f < 0):
            raise Rejected("negative profile on sampled positive radius")
        if derivative:
            D = np.concatenate(derivatives).reshape(shape)
            if not np.all(np.isfinite(D)):
                raise Unresolved("nonfinite logarithmic derivative")
            return f, D
        return f

def _mesh(ctx, extent, step, phase):
    # Endpoint-inclusive trapezoid and a distinct midpoint-phase mesh.
    n = int(math.ceil(2*extent/step))
    edges = np.linspace(-extent,extent,n+1)
    t = edges if phase == 0 else (edges[:-1]+edges[1:])/2
    f = ctx.field(t)
    if phase == 0:
        area = np.diff(t)*(f[:-1]+f[1:])/2*(2/math.pi)
    else:
        area = np.diff(edges)*f*(2/math.pi)
    mass = float(math.fsum(area))
    return t,f,area,mass

def _quantile(t,area,mass,q):
    cdf = np.concatenate(([0.],np.cumsum(area)))
    return float(np.interp(q*mass,cdf,t))

def survey_core(ctx):
    rows=[]
    previous=None
    for extent in ctx.cfg.extents:
        mass=0.; mesh_error=float("inf")
        for refinement in range(ctx.cfg.survey_refinements):
            step=ctx.cfg.survey_step/(2**refinement)
            t,f,area,mass=_mesh(ctx,extent,step,0)
            _,_,_,phase_mass=_mesh(ctx,extent,step,0.5)
            _,_,_,fine_mass=_mesh(ctx,extent,step/2,0)
            mesh_error=max(abs(mass-fine_mass),abs(phase_mass-fine_mass))
            if mesh_error <= ctx.cfg.mass_tolerance:
                t,f,area,mass=_mesh(ctx,extent,step/2,0)
                break
        shell=(t[:-1] < -extent+4)|(t[1:] > extent-4)
        edge=float(math.fsum(area[shell]))
        domain=None if previous is None else abs(mass-previous)
        inner_error=None
        if mass>0 and abs(mass-1)<=ctx.cfg.gauge_tolerance and mesh_error<=ctx.cfg.mass_tolerance:
            _,_,_,inner_mass=_mesh(ctx,extent-ctx.cfg.operator_extension,step/2,0)
            inner_error=abs(mass-inner_mass)
        rows.append(dict(extent=extent,mass=mass,mesh_error=mesh_error,
                         phase_mass=phase_mass,domain_error=domain,inner_domain_error=inner_error,outer_shell_mass=edge,
                         step=step/2,zero_samples=int(np.sum(f==0))))
        if (mass>0 and ((domain is not None and domain<=ctx.cfg.mass_tolerance) or
                           (inner_error is not None and inner_error<=ctx.cfg.mass_tolerance))
            and edge<=ctx.cfg.mass_tolerance and mesh_error<=ctx.cfg.mass_tolerance):
            center_uncertainty=mesh_error+(inner_error if inner_error is not None else domain)+edge
            if mass<1-ctx.cfg.gauge_tolerance:
                previous=mass
                continue  # Search all declared extents before calling missing mass unresolved.
            if mass>1+ctx.cfg.gauge_tolerance:
                if mass-center_uncertainty>1+ctx.cfg.gauge_tolerance:
                    raise Rejected("center overshoot beyond empirical quadrature uncertainty")
                raise Unresolved("canonical center interval unresolved")
            lo=_quantile(t,area,mass,ctx.cfg.quantile_lower)
            hi=_quantile(t,area,mass,ctx.cfg.quantile_upper)
            # Large slope changes of normalization density trigger extra sites.
            curvature=np.abs(np.diff(f,n=2))
            chosen=np.argsort(curvature,kind="stable")[-ctx.cfg.transition_sites:]+1 if ctx.cfg.transition_sites else []
            transitions=t[chosen]
            transitions=transitions[(transitions>=lo-ctx.cfg.guard)&(transitions<=hi+ctx.cfg.guard)]
            return dict(status="resolved",rows=rows,center_mass=mass,
                        core_log_bounds=[lo,hi],domain=[-extent,extent],
                        center_error=abs(mass-1),mesh_error=mesh_error,
                        domain_error=inner_error if inner_error is not None else domain,outer_shell_mass=edge,
                        transitions=transitions.tolist(),
                        quantiles={str(q):_quantile(t,area,mass,q) for q in (.001,.01,.1,.5,.9,.99,.999)})
        previous=mass
    raise Unresolved("normalization mass/domain/mesh unresolved: "+repr(rows))

def _K(d):
    q=np.exp(-np.abs(d))
    with np.errstate(divide="ignore"):
        return np.where(np.abs(d)<0.01,
                        np.log1p(q)-np.log(-np.expm1(-np.abs(d))),
                        np.log1p(q)-np.log1p(-q))

@lru_cache(maxsize=16)
def _legendre_rule(order):
    points,weights=np.polynomial.legendre.leggauss(order)
    points.flags.writeable=False
    weights.flags.writeable=False
    return points,weights

def _panels(lo,hi,pole,width,order):
    # Resolve the pole at a panel boundary and map each adjacent panel by s^4.
    lo,hi,pole=float(lo),float(hi),float(pole)
    if lo<pole<hi:
        # Anchor panels to the pole: an arbitrarily short pole split must not
        # leave an adjacent ordinary panel with a near-endpoint log singularity.
        left=pole-width*np.arange(1,int(math.ceil((pole-lo)/width))+1,dtype=float)
        right=pole+width*np.arange(1,int(math.ceil((hi-pole)/width))+1,dtype=float)
        boundaries=np.sort(np.r_[lo,left[left>lo],pole,right[right<hi],hi])
    else:
        boundaries=np.linspace(lo,hi,max(2,int(math.ceil((hi-lo)/width))+1))
    points,weights=_legendre_rule(order)
    u=(points+1)/2
    w=weights/2
    aa=boundaries[:-1,None]
    bb=boundaries[1:,None]
    t=aa+(bb-aa)*u
    wt=(bb-aa)*np.broadcast_to(w,t.shape)
    right=np.isclose(aa[:,0],pole,rtol=0,atol=1e-14)
    left=np.isclose(bb[:,0],pole,rtol=0,atol=1e-14)
    offset=t-pole
    offset[right]=(bb[right]-aa[right])*u**4
    offset[left]=-(bb[left]-aa[left])*u**4
    t[right]=pole+offset[right]
    wt[right]=(bb[right]-aa[right])*4*u**3*w
    t[left]=pole+offset[left]
    wt[left]=(bb[left]-aa[left])*4*u**3*w
    return t.ravel(),wt.ravel(),offset.ravel()

def operator_values(ctx,sites,domain,order):
    """Return g and W=V/y with panel pole subtraction and finite-bound correction."""
    lo,hi=domain
    output=[]
    for start in range(0,len(sites),ctx.cfg.batch_sites):
        ctx.check()
        chunk=sites[start:start+ctx.cfg.batch_sites]
        meshes=[_panels(lo,hi,float(p),ctx.cfg.operator_panel_width,order) for p in chunk]
        n=max(len(t) for t,w,d in meshes)
        # Different pole insertions can change node count by one panel.
        ts=np.full((len(chunk),n),float(lo),dtype=np.float64)
        ws=np.zeros_like(ts,dtype=np.float64)
        offsets=np.ones_like(ts,dtype=np.float64)
        for i,(t,w,d) in enumerate(meshes):
            ts[i,:len(t)],ws[i,:len(w)],offsets[i,:len(d)]=t,w,d
        fx=ctx.field(ts)
        fy,D=ctx.field(chunk,derivative=True)
        if np.any(fy==0) or np.any((np.abs(fy)>0)&(np.abs(fy)<np.finfo(float).tiny)):
            raise Unresolved("inactive/subnormal profile at assessment site")
        d=offsets
        q=np.exp(-np.abs(d))
        K=_K(d)
        pos=d>=0
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            sub=np.where(pos,(fx-q*fy[:,None])/(-np.expm1(-2*d)),
                         (q*q*fx-q*fy[:,None])/np.expm1(2*d))
            kernel=np.where(pos,K/q,q*K)
        close=np.abs(d)<1e-7
        raw_sub=sub.copy()
        limit=(fy[:,None]+D[:,None])/2
        # Independent local curvature and patch threshold comparisons account for
        # the approximation shared by global order/domain refinements. Estimates
        # are empirical: unrestricted unsampled spikes cannot be certified.
        h=1e-4
        F2=(ctx.field(chunk+h)-2*fy+ctx.field(chunk-h))/(h*h)
        F2_half=(ctx.field(chunk+h/2)-2*fy+ctx.field(chunk-h/2))/(h*h/4)
        curvature_error=np.abs(F2_half-F2)+16*np.finfo(float).eps*np.abs(fy)/(h*h/4)
        taylor_error=np.abs(d)/4*(np.abs(fy[:,None])+2*np.abs(D[:,None])
                                      +np.abs(F2_half[:,None])+curvature_error[:,None])
        alternate=np.where(np.abs(d)<1e-9,limit,raw_sub)
        patch_error=np.where(close,np.abs(alternate-limit)+taylor_error,0)
        patch_error=np.where(np.isfinite(patch_error),patch_error,math.inf)
        pole_error=np.sum(np.abs(ws)*patch_error,axis=1)*(2/math.pi)
        sub=np.where(close,limit,sub)
        kernel=np.where(q==0,np.where(pos,2.,0.),kernel)
        padding=ws==0
        sub[padding]=0
        kernel[padding]=0
        if not np.all(np.isfinite(sub)) or not np.all(np.isfinite(kernel)):
            raise Unresolved("nonfinite nonlocal quadrature kernel")
        corr=fy/2*(_K(lo-chunk)-_K(hi-chunk))
        g=(np.sum(ws*sub,axis=1)+corr)*(2/math.pi)
        W=np.sum(ws*kernel*fx,axis=1)/math.pi
        magnitude_g=(np.sum(np.abs(ws*sub),axis=1)+np.abs(corr))*(2/math.pi)
        magnitude_W=np.sum(np.abs(ws*kernel*fx),axis=1)/math.pi
        eps=64*np.finfo(float).eps
        # Floating summation estimate, explicitly not a rigorous continuum bound.
        eg=eps*magnitude_g+pole_error
        eW=eps*magnitude_W
        output.append(np.stack((g,W,eg,eW,pole_error),axis=1))
        ctx.counters["operator_calls"]+=1
        ctx.counters["operator_sites"]+=len(chunk)
        ctx.counters["operator_nodes"]+=sum(len(t) for t,w,d in meshes)
    return np.concatenate(output)

def stable_terms(f,D,g,W,beta,a,eg=0.,eW=0.):
    """Assemble signed products in logs before common rescaling or squaring."""
    arrays=np.broadcast_arrays(f,D,g,W,eg,eW)
    f,D,g,W,eg,eW=arrays
    if not np.all(np.isfinite(eg)) or not np.all(np.isfinite(eW)) or np.any(eg<0) or np.any(eW<0):
        raise Unresolved("nonfinite or negative operator uncertainty")
    lam=(1+beta)/(1-a)
    factors=((f,), (np.full_like(f,beta),D),
             (np.full_like(f,a*lam),W,D), (np.full_like(f,-lam),g,f))
    logs=[];signs=[]
    for term in factors:
        sign=np.ones_like(f)
        log=np.zeros_like(f)
        zero=np.zeros_like(f,dtype=bool)
        for x in term:
            if not np.all(np.isfinite(x)):
                raise Unresolved("nonfinite equation factor")
            if np.any((x!=0)&(np.abs(x)<np.finfo(float).tiny)):
                raise Unresolved("subnormal equation factor")
            sign*=np.sign(x)
            zero|=x==0
            with np.errstate(divide="ignore"):
                log+=np.log(np.abs(x))
        logs.append(np.where(zero,-np.inf,log))
        signs.append(sign)
    logs=np.stack(logs)
    maximum=np.max(logs,axis=0)
    if not np.all(np.isfinite(maximum)):
        raise Unresolved("all-zero equation terms")
    scaled=np.stack(signs)*np.exp(logs-maximum)
    # fsum avoids catastrophic accumulation for four nearly cancelling terms.
    residual=np.array([math.fsum(scaled[:,i]) for i in range(f.size)]).reshape(f.shape)
    den=np.sum(scaled*scaled,axis=0)
    ratio=residual*residual/den
    def scaled_product_error(coefficient,x,error):
        with np.errstate(divide="ignore",invalid="ignore"):
            z=np.log(abs(coefficient))+np.log(np.abs(x))+np.log(error)-maximum
        return np.where((x==0)|(error==0),0,np.exp(z))
    term_errors=32*np.finfo(float).eps*np.abs(scaled)
    term_errors[2]+=scaled_product_error(a*lam,D,eW)
    term_errors[3]+=scaled_product_error(lam,f,eg)
    if not np.all(np.isfinite(term_errors)):
        raise Unresolved("nonfinite scaled term uncertainty")
    error=np.sum(term_errors,axis=0)
    norm_lower=np.sum(np.maximum(0,np.abs(scaled)-term_errors)**2,axis=0)
    norm_upper=np.sum((np.abs(scaled)+term_errors)**2,axis=0)
    if np.any(norm_lower<=0) or not np.all(np.isfinite(norm_upper)):
        raise Unresolved("term-vector denominator interval reaches zero or nonfinite")
    lower=np.maximum(0,np.abs(residual)-error)**2/norm_upper
    upper=(np.abs(residual)+error)**2/norm_lower
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise Unresolved("nonfinite relative residual interval")
    return dict(ratio=ratio,lower=lower,upper=upper,error=error,
                norm_lower=norm_lower,norm_upper=norm_upper,term_errors=term_errors,
                maximum_log_term=maximum,scaled_terms=scaled,
                roundoff_limited=np.abs(residual)<=error)

def _pure_checks(ctx,sites):
    zero=torch.zeros(1,dtype=torch.float64,requires_grad=True)
    ctx.check()
    value=ctx.family(ctx.anchor,zero)
    if not isinstance(value,torch.Tensor) or value.shape!=zero.shape or not value.requires_grad:
        raise Rejected("origin requires differentiable shape-preserving profile")
    slope=torch.autograd.grad(value.sum(),zero)[0]
    origin=float(value.detach()); slope=float(slope.detach())
    ctx.counters["field_calls"]+=1;ctx.counters["field_elements"]+=1;ctx.counters["derivative_calls"]+=1
    if not math.isfinite(origin) or not math.isfinite(slope):
        raise Unresolved("nonfinite origin gauge")
    if abs(origin)>ctx.cfg.gauge_tolerance or abs(slope-1)>ctx.cfg.gauge_tolerance:
        raise Rejected("origin/slope canonical gauge violation")
    sampled=sites[::max(1,len(sites)//17)]
    f=ctx.field(sampled); again=ctx.field(sampled)
    negative=ctx.field(sampled,signed=True)
    single=np.array([ctx.field(np.array([t]))[0] for t in sampled])
    scale=np.maximum(np.abs(f),np.finfo(float).tiny)
    repeats=float(np.max(np.abs(f-again)/scale))
    batches=float(np.max(np.abs(f-single)/scale))
    odd=float(np.max(np.abs(f+negative)/scale))
    if repeats>1e-12 or batches>1e-10:
        raise Rejected("profile depends on repeat/batch context")
    if odd>ctx.cfg.gauge_tolerance:
        raise Rejected("sampled oddness violation")
    if np.any(f<=0):
        raise Unresolved("full-support positivity unresolved")
    return dict(origin_value=origin,origin_slope=slope,oddness_relative=odd,
                repeatability_relative=repeats,singleton_batch_relative=batches)

def _mean(t,values):
    return float(np.trapezoid(values,t)/(t[-1]-t[0]))

def _coverage_finish(ctx, beta, fine, f, D, domain, extended, values, final, refinements):
    differences=np.maximum(np.abs(values[-1][:,:2]-values[-2][:,:2]),
                           np.abs(values[-2][:,:2]-values[-3][:,:2]))
    domain_difference=np.abs(final[:,:2]-values[-1][:,:2])
    eg=differences[:,0]+domain_difference[:,0]+final[:,2]
    eW=differences[:,1]+domain_difference[:,1]+final[:,3]
    final_terms=stable_terms(f,D,final[:,0],final[:,1],beta,ctx.anchor,eg,eW)
    width=final_terms["upper"]-final_terms["lower"]
    ctx.details.update(operator_errors=dict(g_mesh_max=float(np.max(differences[:,0])),
                       W_mesh_max=float(np.max(differences[:,1])),
                       g_domain_max=float(np.max(domain_difference[:,0])),
                       W_domain_max=float(np.max(domain_difference[:,1])),
                       maximum_residual_interval_width=float(np.max(width)),
                       pole_limit_error_max=float(np.max(final[:,4]))))
    # Qualification is resolution of the equation measurement; no target residual cutoff.
    if float(np.max(width))>ctx.cfg.operator_tolerance:
        raise Unresolved("nonlocal equation uncertainty exceeds fixed tolerance")

    return dict(differences=differences, domain_difference=domain_difference,
                eg=eg, eW=eW, final_terms=final_terms, refinements=refinements)

def _coverage_bundle(ctx, beta, fine, survey, lo, hi, coverage_count=None):
    # The original scientific checks and quadrature order are retained.
    if coverage_count is None:
        coverage_count = len(fine)
    checks=_pure_checks(ctx,fine)
    ctx.details.update(checks=checks,coverage_sites=coverage_count,assessment_log_bounds=[lo,hi])
    f,D=ctx.field(fine,derivative=True)
    h=1e-4
    fd=(ctx.field(fine+h)-ctx.field(fine-h))/(2*h)
    fd_half=(ctx.field(fine+h/2)-ctx.field(fine-h/2))/h
    derivative_scale=np.maximum(np.abs(f),np.abs(D))
    derivative_error=float(np.max(np.abs(D-fd_half)/derivative_scale))
    derivative_mesh_error=float(np.max(np.abs(fd_half-fd)/derivative_scale))
    checks.update(derivative_relative_error=derivative_error,derivative_mesh_error=derivative_mesh_error)
    if derivative_error>ctx.cfg.score_tolerance or derivative_mesh_error>ctx.cfg.score_tolerance:
        raise Unresolved("autograd/log finite-difference derivative agreement unresolved")
    negative=ctx.field(fine,signed=True)
    if np.any(f<=0):
        raise Unresolved("inactive full-support profile in core/transition coverage")
    if np.max(np.abs(f+negative)/f)>ctx.cfg.gauge_tolerance:
        raise Rejected("oddness violation across core/transition coverage")
    domain=survey["domain"]
    if lo<=domain[0] or hi>=domain[1]:
        raise Unresolved("core guard shells not inside resolved operator domain")
    all_measured = []
    ctx.details['operator_refinement_sites'] = len(fine)
    refinements=[]
    ctx.details["operator_refinements"]=refinements
    values=[]
    for order in ctx.cfg.operator_orders:
        measured=operator_values(ctx,fine,domain,order)
        values.append(measured)
        all_measured.append(measured)
        terms=stable_terms(f,D,measured[:,0],measured[:,1],beta,ctx.anchor,measured[:,2],measured[:,3])
        refinements.append(dict(order=order,mean_square=_mean(fine,terms["ratio"]),
                                worst_ratio=float(np.max(terms["ratio"]))))
    extended=[domain[0]-ctx.cfg.operator_extension,domain[1]+ctx.cfg.operator_extension]
    final=operator_values(ctx,fine,extended,ctx.cfg.operator_orders[-1])

    result = _coverage_finish(ctx, beta, fine, f, D, domain, extended,
                              values, final, refinements)
    result.update(f=f, D=D, checks=checks, domain=domain, extended=extended,
                  measured=all_measured, final=final)
    return result

def _merge_coverage_bundle(ctx, beta, old_sites, new_sites, old, new):
    # This is only a merge of already evaluated pointwise quantities.
    joined = np.r_[old_sites, new_sites]
    order = np.argsort(joined, kind='stable')
    fine = joined[order]
    f = np.concatenate((old['f'], new['f']))[order]
    D = np.concatenate((old['D'], new['D']))[order]
    final = np.concatenate((old['final'], new['final']), axis=0)[order]
    measured = [np.concatenate((x, y), axis=0)[order]
                for x, y in zip(old['measured'], new['measured'])]
    checks = dict(old['checks'])
    maxima = ('repeatability_relative', 'oddness_relative',
              'singleton_batch_relative', 'derivative_relative_error',
              'derivative_mesh_error')
    for key in maxima:
        if key in new['checks']:
            checks[key] = max(checks.get(key, 0.0), new['checks'][key])
    refinements = []
    ctx.details.update(checks=checks, coverage_sites=len(fine),
                       operator_refinements=refinements,
                       operator_refinement_sites=len(fine))
    for quadrature_order, values in zip(ctx.cfg.operator_orders, measured):
        terms = stable_terms(f, D, values[:, 0], values[:, 1], beta,
                             ctx.anchor, values[:, 2], values[:, 3])
        refinements.append(dict(order=quadrature_order,
                                mean_square=_mean(fine, terms['ratio']),
                                worst_ratio=float(np.max(terms['ratio']))))
    result = _coverage_finish(ctx, beta, fine, f, D, old['domain'],
                              old['extended'], [x[:, :2] for x in measured],
                              final, refinements)
    result.update(f=f, D=D, checks=checks, domain=old['domain'],
                  extended=old['extended'], measured=measured, final=final)
    return fine, result

def assess_anchor(ctx):
    ctx.check()
    beta_tensor=ctx.family.beta(ctx.anchor)
    ctx.counters["beta_calls"]+=1
    if not isinstance(beta_tensor,torch.Tensor) or beta_tensor.numel()!=1:
        raise Rejected("beta must be a scalar Torch tensor")
    beta=float(beta_tensor.detach())
    if not math.isfinite(beta):
        raise Unresolved("nonfinite beta")
    if beta<=0:
        raise Rejected("nonpositive beta outside target focusing branch")
    survey=survey_core(ctx)
    ctx.details.update(beta=beta,survey=survey,core_log_bounds=survey["core_log_bounds"])
    core=survey["core_log_bounds"]
    lo,hi=core[0]-ctx.cfg.guard,core[1]+ctx.cfg.guard
    base=np.unique(np.r_[np.linspace(lo,hi,ctx.cfg.coverage_sites),
                          survey["transitions"],list(survey["quantiles"].values())])
    fine=np.unique(np.r_[base,(base[:-1]+base[1:])/2])

    coverage_history = []
    ctx.details['coverage_refinements'] = coverage_history
    old_sites = new_sites = previous = None
    for level in range(2):  # Initial comparison plus at most ONE extra midpoint level.
        record = dict(level=level, coarse_sites=len(base), fine_sites=len(fine),
                      coarse_mean=None, fine_mean=None, absolute_change=None,
                      coarse_mean_interval=None, fine_mean_interval=None,
                      tolerance=ctx.cfg.score_tolerance, status='evaluating')
        coverage_history.append(record)
        ctx.details['coverage_sites'] = len(fine)
        try:
            ctx.check()
            if level == 0:
                bundle = _coverage_bundle(ctx, beta, fine, survey, lo, hi)
            else:
                new = _coverage_bundle(ctx, beta, new_sites, survey, lo, hi,
                                       coverage_count=len(fine))
                fine, bundle = _merge_coverage_bundle(ctx, beta, old_sites,
                                                      new_sites, previous, new)
        except Exception as exc:
            record['status'] = 'interrupted'
            record['reason'] = str(exc)
            raise
        final_terms = bundle['final_terms']
        indices = np.searchsorted(fine, base)
        mean = _mean(fine, final_terms['ratio'])
        coarse_mean = _mean(base, final_terms['ratio'][indices])
        record.update(coarse_mean=coarse_mean, fine_mean=mean,
                      absolute_change=abs(mean-coarse_mean),
                      coarse_mean_interval=[_mean(base, final_terms['lower'][indices]),
                                            _mean(base, final_terms['upper'][indices])],
                      fine_mean_interval=[_mean(fine, final_terms['lower']),
                                          _mean(fine, final_terms['upper'])],
                      status='measured')
        # All numbers are retained BEFORE a convergence or refinement decision.
        if abs(mean-coarse_mean) <= ctx.cfg.score_tolerance:
            record['status'] = 'qualified'
            break
        if level == 1:
            record['status'] = 'unresolved'
            raise Unresolved('residual coverage refinement unresolved')
        record['status'] = 'refinement_requested'
        old_sites = fine
        previous = bundle
        base = fine
        new_sites = np.setdiff1d((fine[:-1]+fine[1:])/2, fine)
        if len(new_sites) == 0:
            record['status'] = 'unresolved'
            raise Unresolved('residual coverage subdivision unrepresentable')
        fine = np.sort(np.r_[old_sites, new_sites])
    f, D = bundle['f'], bundle['D']
    checks = bundle['checks']
    domain, extended = bundle['domain'], bundle['extended']
    final = bundle['final']
    differences, domain_difference = bundle['differences'], bundle['domain_difference']
    refinements = bundle['refinements']
    width = final_terms['upper'] - final_terms['lower']
    fixed=np.linspace(math.log(1e-3),math.log(1e3),ctx.cfg.fixed_sites)
    legacy=dict(status="unresolved",score=None)
    try:
        ff,DD=ctx.field(fixed,derivative=True)
        op=operator_values(ctx,fixed,extended,ctx.cfg.operator_orders[-1])
        tt=stable_terms(ff,DD,op[:,0],op[:,1],beta,ctx.anchor,op[:,2],op[:,3])
        # Reconstruction used only for diagnosing the clamp, never ranking.
        with np.errstate(over="ignore",under="ignore"):
            actual=tt["scaled_terms"]*np.exp(tt["maximum_log_term"])
            old=np.sum(actual,axis=0)**2/np.maximum(np.sum(actual**2,axis=0),1e-28)
        legacy=dict(status="diagnostic_only",stable_mean_square=float(np.mean(tt["ratio"])),
                    clamped_mean_square=float(np.mean(old)) if np.all(np.isfinite(old)) else None,
                    term_log_scale_min=float(np.min(tt["maximum_log_term"])))
    except Unresolved as error:
        legacy["reason"]=str(error)
    error_gauge=checks["origin_value"]**2+(checks["origin_slope"]-1)**2+survey["center_error"]**2
    return dict(a=ctx.anchor,status="qualified",score=mean+error_gauge,beta=beta,
                residual_mean_square=mean,residual_rms=math.sqrt(mean),
                residual_worst_ratio=float(np.max(final_terms["ratio"])),
                residual_mean_square_interval=[_mean(fine,final_terms["lower"]),_mean(fine,final_terms["upper"])],
                residual_mesh_mean_square_change=abs(mean-coarse_mean),
                roundoff_limited_sites=int(np.sum(final_terms["roundoff_limited"])),
                core_log_bounds=core,assessment_log_bounds=[lo,hi],
                coverage_sites=len(fine),
                coverage_refinements=coverage_history,survey=survey,checks=checks,
                scales=dict(profile_max=float(np.max(f)),D_max=float(np.max(np.abs(D))),
                            g_max=float(np.max(np.abs(final[:,0]))),W_max=float(np.max(np.abs(final[:,1]))),
                            term_log_min=float(np.min(final_terms["maximum_log_term"])),
                            term_log_max=float(np.max(final_terms["maximum_log_term"]))),
                operators=dict(method="pole-subtracted core-domain log panels, quartic pole mapping",
                               refinements=refinements,domain=domain,extended_domain=extended,
                               g_mesh_max=float(np.max(differences[:,0])),
                               W_mesh_max=float(np.max(differences[:,1])),
                               g_domain_max=float(np.max(domain_difference[:,0])),
                               W_domain_max=float(np.max(domain_difference[:,1])),
                               maximum_residual_interval_width=float(np.max(width)),
                               pole_limit_error_max=float(np.max(final[:,4])),
                               uncertainty_interpretation="estimated mesh/domain/local-patch errors, not rigorous bounds"),
                fixed_window=legacy)

def assess_family(family, *, anchors=PUBLIC_ANCHORS, config=None, budget=None):
    started=time.monotonic();counters=Counter();rows=[];reason=None
    cfg=_config(config)
    anchors=tuple(float(a) for a in anchors)
    if not anchors or any(a not in PUBLIC_ANCHORS for a in anchors) or len(set(anchors))!=len(anchors):
        raise ValueError("Common assessor accepts distinct declared public anchors only")
    snapshots={key:value.detach().clone() for key,value in family.state_dict().items()}
    status="qualified"
    try:
        for value in list(family.parameters())+list(family.buffers()):
            if value.device.type!="cpu" or (value.dtype.is_floating_point and value.dtype!=torch.float64):
                raise Rejected("family state must use CPU float64")
        for a in anchors:
            ctx=Context(family,a,cfg,budget,counters)
            try:
                rows.append(assess_anchor(ctx))
            except Rejected as error:
                rows.append(dict(a=a,status="rejected",score=None,reason=str(error),**ctx.details))
                status="rejected";reason=str(error);break
            except (Unresolved,TimeoutError,OverflowError,FloatingPointError) as error:
                rows.append(dict(a=a,status="unresolved",score=None,reason=str(error),**ctx.details))
                status="unresolved";reason=str(error);break
    except Rejected as error:
        status="rejected";reason=str(error)
    except Exception as error:
        # No failure or partial produces a ranking score.
        status="unresolved";reason=type(error).__name__+": "+str(error)
    state=family.state_dict()
    changed=list(state)!=list(snapshots) or any(not torch.equal(state[k],v) for k,v in snapshots.items() if k in state)
    if changed:
        status="rejected";reason="candidate mutated parameter/buffer state during assessment"
    score=float(math.fsum(r["score"] for r in rows)/len(rows)) if status=="qualified" and len(rows)==len(anchors) else None
    if score is not None and not math.isfinite(score):
        status="unresolved";reason="nonfinite aggregate score";score=None
    result = dict(version=VERSION,status=status,score=score,anchors=rows,counters=dict(counters),
                diagnostics=dict(reason=reason,elapsed_seconds=time.monotonic()-started,
                                 immutable_state=not changed,config=asdict(cfg),
                                 core_log_bounds={str(r["a"]):r.get("core_log_bounds") for r in rows},
                                 interpretation="Finite numerical resolution; no continuum/reference/branch uniqueness certificate"))

    nonfinite_paths=[]
    def json_safe(value,path=""):
        if isinstance(value,(float,np.floating)) and not math.isfinite(value):
            nonfinite_paths.append(path)
            return None
        if isinstance(value,dict):
            return {k:json_safe(v,path+"."+str(k)) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            return [json_safe(v,path+"."+str(i)) for i,v in enumerate(value)]
        return value
    result=json_safe(result)
    result["diagnostics"]["nonfinite_measurement_paths"]=nonfinite_paths
    return result
