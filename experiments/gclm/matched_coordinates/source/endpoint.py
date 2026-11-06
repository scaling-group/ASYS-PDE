"""Analytic tail coefficients from the saved profile (80-digit arithmetic)."""
import math
import mpmath as mp
import torch

def endpoint_algebra(raw,family):
    """Analytic Legendre moments and endpoint derivatives; 80-digit arithmetic."""
    mp.mp.dps=80
    d=[sum(mp.mpf(float(t))*mp.mpf(float(c)) for t,c in zip(row,raw.correction.detach()))
       for row in raw.coefficient_transport]
    if family=='poly':
        q0=sum(d[j]*(-1)**j for j in range(14))
        q1=sum(d[j]*2*(-1)**(j+1)*j*j for j in range(1,14))
        def evaluate(v):return sum(d[j]*mp.chebyt(j,2*v-1) for j in range(14))
    else:
        # Match the saved binary64 beta/exponent used by the native evaluator.
        p=mp.mpf(float(2*raw.beta()))
        def moment(q,j):
            return mp.sqrt(2*j+1)*mp.fprod(q-k for k in range(j))/mp.fprod(q+k+1 for k in range(j+1))
        def F(q,v=None,derivative=False):
            scale=mp.fprod((q-j)/(q+j+1) for j in range(8))
            if v is None:
                # Every q>1: the value and first derivative of v**q vanish.
                assert q>1
                endpoint=lambda j: (-1)**(j+1)*j*(j+1)*mp.sqrt(2*j+1) if derivative else (-1)**j*mp.sqrt(2*j+1)
                return -sum(moment(q,j)*endpoint(j) for j in range(8))/scale
            return (v**q-sum(moment(q,j)*mp.sqrt(2*j+1)*mp.legendre(j,2*v-1) for j in range(8)))/scale
        residues=[mp.fprod(2*p+j+k+1 for k in range(3))/mp.fprod(j-k for k in range(3) if j!=k) for j in range(3)]
        def Q(v=None,derivative=False):
            fp,fq,fs,ft=[F(q,v,derivative) for q in (p,p+1,p+2,2*p)]
            u=mp.sqrt(2*p+1)*fp
            w=mp.sqrt(2*p+3)*((2*p+2)*fq-(2*p+1)*fp)
            z=mp.sqrt(2*p+5)*((2*p+3)*(p+2)*fs-2*(p+1)*(2*p+3)*fq+(2*p+1)*(p+1)*fp)
            e=mp.sqrt(4*p+1)*(ft+sum(residues[j]*(ft-f)/(p-j) for j,f in enumerate((fp,fq,fs))))
            if v is None:
                polynomial=sum(d[j]*((-1)**(j+1)*j*(j+1) if derivative else (-1)**j)*mp.sqrt(2*j+1) for j in range(8))
                rational=0
            else:
                polynomial=sum(d[j]*mp.sqrt(2*j+1)*mp.legendre(j,2*v-1) for j in range(8))
                R=v*v*(1-v)**2/((v-mp.mpf('.5'))**2+mp.mpf(1)/64)
                rational=(d[12]+(2*v-1)*d[13])*R
            return polynomial+sum(d[8+j]*f for j,f in enumerate((u,w,z,e)))+rational
        q0,q1=Q(),Q(derivative=True)
        evaluate=Q
        # Independent native endpoint formula from its stable forward assembly.
        native=sum(d[j]*(-1)**j*mp.sqrt(2*j+1) for j in range(8))+8/p*(d[8]*mp.sqrt(2*p+1)-d[9]*mp.sqrt(2*p+3)+d[10]*mp.sqrt(2*p+5))-4*(p+3)/(p*p)*d[11]*mp.sqrt(4*p+1)
        assert abs(q0-native)<mp.mpf('1e-60')
    discrepancies=[]
    for v in (.05,.2,.5,.8):
        # Compare algebra to saved native forward at ordinary interior points.
        x=math.sqrt(v**(-2*float(raw.beta()))-1)
        t=torch.tensor([x],dtype=torch.float64)
        lv=-math.log(math.hypot(x,1))/float(raw.beta())
        observed=(math.log(float(raw(t)[0])/(x/math.hypot(x,1)))-lv)/(-math.expm1(lv))**2
        expected=float(evaluate(mp.mpf(v)))
        discrepancies.append(abs(observed-expected))
    assert max(discrepancies)<1e-7,discrepancies
    return float(q0),float(q1),max(discrepancies)

