"""Formal recurrence in logarithmic envelope Q=log(6F/x^2).

Native derivation of the higher logarithmic transport; no reference profile.
Q_t means (u*d/du)Q. The scaled equation is
exp(Q)*(6+7Q_t+Q_tt+2Q_t^2)-6-14Q_t=0.
"""
from pathlib import Path
import json
import sympy as s
z,c=s.symbols('log_u resonant_constant')
K=11
q={0:s.Integer(0),1:s.Rational(6,100),2:s.Rational(117,10000),3:s.Rational(4932,1000000),4:s.Rational(727947,2*10**8),5:s.Rational(271803276,5*10**10)}

def forcing(q,k):
    # exp coefficients by formal differentiation in u at fixed log_u.
    e={0:s.Integer(1)}
    for n in range(1,k+1):
        e[n]=s.expand(sum(j*q.get(j,0)*e[n-j] for j in range(1,n+1))/n)
    d={n:s.expand(n*p+s.diff(p,z)) for n,p in q.items()}
    dd={n:s.expand(n*p+s.diff(p,z)) for n,p in d.items()}
    g={n:s.expand(7*d.get(n,0)+dd.get(n,0)+2*sum(d.get(i,0)*d.get(n-i,0) for i in range(n+1))) for n in range(k+1)}
    g[0]+=6
    return s.expand(sum(e[i]*g[k-i] for i in range(k+1))-14*d.get(k,0))

log=[]; constant=[]; transport=[]
lines=[__doc__]
for k in range(6,K+1):
    p=forcing(q,k)
    if k==6:
        q[k]=s.expand(-p/5*z+c)
    else:
        degree=s.degree(p,z)
        a=s.symbols('a:'+str(degree+1))
        trial=sum(v*z**i for i,v in enumerate(a))
        eq=s.expand((k-1)*(k-6)*trial+(2*k-7)*s.diff(trial,z)+s.diff(trial,z,2)+p)
        sol=s.solve([eq.coeff(z,i) for i in range(degree+1)],a)
        q[k]=s.expand(trial.subs(sol))
    lines.append(f'q{k} = {q[k]}')
    log.append(float(q[k].coeff(z,1)))
    constant.append(float(q[k].subs({z:0,c:0})))
    transport.append(float(s.diff(q[k],c).subs(z,0)))
report={'log_coefficients':log,'constant_coefficients':constant,'resonant_transport':transport,'order':K}
print('\n'.join(lines))
print(json.dumps(report,indent=2))

# Re-substitution and the first uncancelled logarithmic obstruction.
checks={k:str(s.simplify(forcing(q,k))) for k in range(1,K+1)}
next_forcing=forcing(q,K+1)
next_lines=['Exact re-substitution through Q order eleven:',str(checks),
            'First forcing, Q order twelve (before adding the smooth remainder):',str(next_forcing),
            'Its log_u^2 coefficient:',str(s.expand(next_forcing).coeff(z,2)),
            'A bounded smooth remainder changes only the constant at this order.',
            'Consequently this construction is not an exact/all-order asymptotic solution.']
assert all(value=='0' for value in checks.values())
print('\n'.join(next_lines))
