"""Quadrature estimate of the forced-PME L1 stability right-hand side.

This evaluates an explicit formula; it neither evolves a PDE nor fits a model.
The estimate is not an interval-certified bound.
"""
import math
import numpy as np
import torch
from diagnostics import derivative, END


def quadrature(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor((a+b)/2 + (b-a)/2*x), torch.tensor((b-a)/2*w)


def front(model, t):
    return float(model.front(t.reshape(1, 1))[0].detach().flatten()[0])


def integral(model, n):
    radius, weights = quadrature(1,6,max(128,n))
    exact = 1.6*(-torch.expm1(-((radius-1)/.6).square()))
    mass0 = float((2*math.pi*radius*weights*exact).sum())
    r0, w0 = quadrature(0,1,max(128,n))
    all_r, all_w = torch.cat((r0,radius)), torch.cat((w0,weights))
    with torch.no_grad():
        initial = model(torch.stack((all_r,torch.zeros_like(all_r)),dim=1)).flatten()
        initial_error = float((2*math.pi*all_r*all_w*(initial-torch.cat((torch.zeros_like(r0),exact))).abs()).sum())
    residual_integral = boundary_integral = 0.
    time_edges = [0,.01,.025,.05,.1,.2,.35,.5,END]
    for ta,tb in zip(time_edges[:-1],time_edges[1:]):
        times, time_weights = quadrature(ta,tb,n)
        for t,tw in zip(times,time_weights):
            edge = front(model,t)
            # Split at the exact analytic front and resolve its shrinking scale.
            lengths = [0,.01*edge,.1*edge,edge,.3,.6,1.5,6-edge]
            edges = sorted(set([edge]+[min(6.,edge+x) for x in lengths]+[6.]))
            rs,ws = zip(*(quadrature(a,b,n) for a,b in zip(edges[:-1],edges[1:]) if b>a))
            r,w = torch.cat(rs),torch.cat(ws)
            points = torch.stack((r,torch.full_like(r,t)),dim=1).requires_grad_(True)
            value = model(points)
            vr,vt = derivative(value,points,0),derivative(value,points,1)
            residual = vt-value*(derivative(vr,points,0)+vr/r[:,None])-vr.square()
            residual_integral += float((2*math.pi*r*w*residual[:,0].abs()).sum().detach())*float(tw)
            boundary = torch.tensor([[6.,float(t)]],requires_grad=True)
            bv=model(boundary); br=derivative(bv,boundary,0)
            boundary_integral += float((2*math.pi*6*(bv*br).abs()).detach())*float(tw)
    return {'gauss_nodes_per_subinterval':n,'exact_initial_pressure_mass':mass0,'initial_l1_error':initial_error,'spacetime_l1_residual':residual_integral,'integrated_outer_flux_mismatch':boundary_integral,'relative_l1_stability_rhs':(initial_error+residual_integral+boundary_integral)/mass0,'analytic_final_front':front(model,torch.tensor(END))}


