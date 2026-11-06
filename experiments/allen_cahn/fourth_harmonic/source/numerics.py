"""Independent diffuse-PDE ETDRK4 and geometric contour measurement; no ASYS model."""
import numpy as np
from scipy.fft import rfft2,irfft2
from scipy.spatial import cKDTree
from scipy.ndimage import spline_filter,map_coordinates
R0=.28

def curve(phi,a):
    norm=np.sqrt(1+a*a/2);r=R0*(1+a*np.cos(2*phi))/norm
    rp=-2*R0*a*np.sin(2*phi)/norm;rpp=-4*R0*a*np.cos(2*phi)/norm
    c,s=np.cos(phi),np.sin(phi)
    x,y=r*c,r*s;xp,yp=rp*c-r*s,rp*s+r*c
    xpp,ypp=(rpp-r)*c-2*rp*s,(rpp-r)*s+2*rp*c
    return x,y,xp,yp,xpp,ypp

def signed_distance(n,a,samples=16384):
    axis=-1+2*np.arange(n)/n;x,y=np.meshgrid(axis,axis,indexing='ij')
    rho=np.hypot(x,y);angle=np.arctan2(y,x)
    if a==0:return R0-rho,dict(method='exact circle distance',projection_stationarity_max=0.)
    ph=2*np.pi*np.arange(samples)/samples;cx,cy,*_=curve(ph,a)
    _,idx=cKDTree(np.column_stack((cx,cy))).query(np.column_stack((x.ravel(),y.ravel())),workers=1)
    ph=ph[idx].reshape(n,n)
    for _ in range(12):
        cx,cy,xp,yp,xpp,ypp=curve(ph,a)
        f=(cx-x)*xp+(cy-y)*yp;fp=xp*xp+yp*yp+(cx-x)*xpp+(cy-y)*ypp
        delta=np.divide(f,fp,out=np.zeros_like(f),where=abs(fp)>1e-15)
        ph-=np.clip(delta,-.01,.01)
    cx,cy,xp,yp,*_=curve(ph,a);d=np.hypot(cx-x,cy-y)
    boundary=R0*(1+a*np.cos(2*angle))/np.sqrt(1+a*a/2)
    d=np.where(rho<=boundary,d,-d)
    near=abs(d)<.12
    return d,dict(method='global nearest sampled contour point, then continuous Euclidean normal projection by Newton; exact parametric curve',seed_samples=samples,projection_stationarity_max=float(np.max(abs((cx-x)*xp+(cy-y)*yp)[near])))

class ETDRK4:
    """Same v=u+1, contour32 coefficients and step algebra as validated source."""
    def __init__(self,u,eps):
        self.n=len(u);self.eps=eps;self.f=rfft2(u+1,workers=1);self.cache={}
        kx=np.fft.fftfreq(self.n,d=2/self.n)*2*np.pi;ky=np.fft.rfftfreq(self.n,d=2/self.n)*2*np.pi
        self.linear=-eps**2*(kx[:,None]**2+ky[None,:]**2)-2
    def coefficients(self,dt):
        key=round(dt,15)
        if key not in self.cache:
            z=dt*self.linear;circle=np.exp(1j*np.pi*(np.arange(32)+.5)/32)
            Q=np.zeros_like(z);f1=Q.copy();f2=Q.copy();f3=Q.copy()
            for r in circle:
                a=z+r;ex=np.exp(a)
                Q+=((np.exp(a/2)-1)/a).real/32
                f1+=((-4-a+ex*(4-3*a+a*a))/a**3).real/32
                f2+=((2+a+ex*(-2+a))/a**3).real/32
                f3+=((-4-3*a-a*a+ex*(4-a))/a**3).real/32
            self.cache[key]=(np.exp(z),np.exp(z/2),dt*Q,dt*f1,dt*f2,dt*f3)
        return self.cache[key]
    def nonlin(self,f):
        val=irfft2(f,s=(self.n,self.n),workers=1)
        return rfft2(val*val*(3-val),workers=1)
    def advance(self,count,dt):
        E,E2,Q,f1,f2,f3=self.coefficients(dt);f=self.f
        for _ in range(count):
            nv=self.nonlin(f)
            a=E2*f+Q*nv;na=self.nonlin(a)
            b=E2*f+Q*na;nb=self.nonlin(b)
            c=E2*a+Q*(2*nb-nv);nc=self.nonlin(c)
            f=E*f+f1*nv+2*f2*(na+nb)+f3*nc
        self.f=f
    def field(self):return irfft2(self.f,s=(self.n,self.n),workers=1)-1

def contour(u,angles=2048,order=5):
    n=len(u);theta=2*np.pi*np.arange(angles)/angles;c,s=np.cos(theta),np.sin(theta)
    coeff=spline_filter(u,order=order,mode='grid-wrap');lo=np.zeros(angles);hi=np.full(angles,.7)
    def val(radius):return map_coordinates(coeff,np.array([(radius*c+1)*n/2,(radius*s+1)*n/2]),order=order,mode='grid-wrap',prefilter=False)
    assert np.all(val(lo)>0) and np.all(val(hi)<0),'Contour is absent or bracket invalid'
    for _ in range(40):
        mid=(lo+hi)/2;inside=val(mid)>0;lo=np.where(inside,mid,lo);hi=np.where(inside,hi,mid)
    radii=(lo+hi)/2
    # Entire polar zero contour; check star-shapedness independently on rays.
    rays=theta[::max(1,angles//64)];rr=np.linspace(0,.7,81)[:,None]
    values=map_coordinates(coeff,np.array([np.broadcast_to((rr*np.cos(rays)+1)*n/2,(81,len(rays))),np.broadcast_to((rr*np.sin(rays)+1)*n/2,(81,len(rays)))]),order=order,mode='grid-wrap',prefilter=False)
    assert np.all(np.sum(np.diff(values>0,axis=0)!=0,axis=0)==1),'Multiple ray crossings'
    r0=float(np.mean(radii));row=dict(r0=r0,area=float(np.pi*np.mean(radii*radii)))
    row['R_area']=np.sqrt(row['area']/np.pi)
    for m in [2,4]:
        row['r'+str(m)]=float(2*np.mean(radii*np.cos(m*theta)))
        row['e'+str(m)]=row['r'+str(m)]/r0
        row['sine'+str(m)]=float(2*np.mean(radii*np.sin(m*theta)))/r0
    return radii,row
