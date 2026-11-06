# Waiting-profile recurrence

For gamma=7/3 and an analytic F=sum(a_n*xi^n), the supplied quadratic/cubic
normalization fixes a2=1/6, a3=1. For n>=4 the provisional recurrence is
[(n-3)(n-8)/6]*a_n + [(n+1)(n+2)/2]*sum_{i=3}^{n-1}(a_i*a_{n+2-i})=0.
It predicts a4=45/2,a5=945,a6=67095,a7=9542610, then zero linear coefficient
at n=8 with positive forcing1034888400. A tentative cancellation uses
-1241866080*xi^8*log(xi), since the derivative of the indicial polynomial
at8 is5/6. Check the signs, coefficients, hypotheses and whether additional
structure is necessary; do not assume this proves nonexistence of other profiles.

A correction with
weaker regularity must still have finite second derivatives and preserve the
required lower traces; a regularization must not be called the strict singular
limit without source-level justification. Normalization C=1 uses the profile
scaling symmetry and is not a physical coefficient inferred from data.
