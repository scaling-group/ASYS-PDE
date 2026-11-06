"""Radial dr dt quadrature, including the clipped r <= 0.5 core."""
import numpy as np

def weights(x,upper=None):
 x=np.asarray(x);assert np.all(np.diff(x)>0);w=np.zeros(len(x))
 for i,L in enumerate(np.diff(x)):
  h=L if upper is None else max(0.,min(float(L),upper-float(x[i])))
  w[i]+=h-h*h/(2*L);w[i+1]+=h*h/(2*L)
 return w

