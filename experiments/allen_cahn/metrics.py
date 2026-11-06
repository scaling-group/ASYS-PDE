"""Full-trajectory field, interface-band and periodic contour metrics."""
import math
import numpy as np
import torch
from scipy.spatial import cKDTree

def weights(t):
    t=np.asarray(t);assert len(t)>1 and np.all(np.diff(t)>0)
    w=np.zeros(len(t));w[:-1]+=np.diff(t)/2;w[1:]+=np.diff(t)/2
    return w


def crossings(u):
    """Linear edge crossings on both periodic grid directions, equal crossing weights."""
    n=len(u);assert u.shape==(n,n);axis=-1+2*np.arange(n)/n;points=[]
    for dim in (0,1):
        v=np.roll(u,-1,axis=dim);i,j=np.where((u>=0)!=(v>=0))
        if len(i):
            fraction=u[i,j]/(u[i,j]-v[i,j]);p=np.column_stack((axis[i],axis[j]))
            p[:,dim]+=2/n*fraction;points.append((p+1)%2-1)
    return np.concatenate(points) if points else np.empty((0,2))


def interface(a,b):
    if not len(a) or not len(b):
        return dict(interface_status='both_empty' if not len(a) and not len(b) else 'one_empty',interface_rms=None,interface_max=None,candidate_crossings=len(a),reference_crossings=len(b))
    a=(a+1)%2;b=(b+1)%2
    ab=cKDTree(b,boxsize=2).query(a,k=1)[0];ba=cKDTree(a,boxsize=2).query(b,k=1)[0]
    return dict(interface_status='present',interface_rms=float(np.sqrt(.5*(np.mean(ab**2)+np.mean(ba**2)))),interface_max=float(max(ab.max(),ba.max())),candidate_crossings=len(a),reference_crossings=len(b))


def energy(a,b):
    ws=(2/len(a))**2;d=a-b;mask=np.abs(b)<=.9
    return dict(E=float(ws*np.sum(d*d)),R=float(ws*np.sum(b*b)),band_E=float(ws*np.sum(d[mask]**2)),band_R=float(ws*np.sum(b[mask]**2)),band_area=float(ws*np.sum(mask)),Linf=float(np.max(np.abs(d))),candidate_positive_area=float(ws*np.sum(a>0)),reference_positive_area=float(ws*np.sum(b>0)))


def pool(rows):
    E=math.fsum(x['weight']*x['E'] for x in rows);R=math.fsum(x['weight']*x['R'] for x in rows)
    bE=math.fsum(x['weight']*x['band_E'] for x in rows);bR=math.fsum(x['weight']*x['band_R'] for x in rows)
    ir=[x for x in rows if x['interface_rms'] is not None];coverage=math.fsum(x['weight'] for x in ir)
    return dict(weighted_E=E,weighted_R=R,field_error_percent=100*math.sqrt(E/R) if R else None,band_error_percent=100*math.sqrt(bE/bR) if bR else None,weighted_band_E=bE,weighted_band_R=bR,interface_time_rms=math.sqrt(math.fsum(x['weight']*x['interface_rms']**2 for x in ir)/coverage) if coverage else None,interface_max=max((x['interface_max'] for x in ir),default=None),interface_time_coverage=coverage,empty_contour_times=len(rows)-len(ir),Linf=max(x['Linf'] for x in rows))


def predict(m,axis,t):
 xx,yy=np.meshgrid(axis,axis,indexing='ij');xy=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,t)]
 with torch.inference_mode():out=np.concatenate([m(torch.tensor(x,dtype=torch.float64)).numpy().ravel() for x in np.array_split(xy,max(1,math.ceil(len(xy)/8192)))])
 return out.reshape(xx.shape)

