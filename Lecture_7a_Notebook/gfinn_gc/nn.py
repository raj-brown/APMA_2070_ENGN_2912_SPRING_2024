#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:40:47 2021

@author: zen
"""
import learner_gfinn as ln
import torch
from learner_gfinn.utils import mse, grad
from learner_gfinn.integrator import RK
    
class GC_S(ln.nn.Module):
    def __init__(self):
        super(GC_S, self).__init__()
        
    def forward(self, x):
        return torch.tensor([[0,0,1,1]], dtype = self.dtype, device = self.device).repeat(x.shape[0], 1)

class GC_E(ln.nn.Module):
    def __init__(self):
        super(GC_E, self).__init__()
        
    def forward(self, x):
        q, p, S1, S2 = x[...,:1], x[...,1:2], x[...,2:3], x[...,3:]
        T1 = (torch.exp(S1) / q) ** (2 / 3) * (2 / 3)
        T2 = (torch.exp(S2) / (2 - q)) ** (2 / 3) * (2 / 3)
        dE = torch.cat([T2/(2-q) -T1/q, p, T1, T2], dim = -1)
        return dE

class gnode_LNN(ln.nn.Module):
    def __init__(self, S, ind, deriv = False):
        super(gnode_LNN, self).__init__()
        self.S = S
        self.ind = ind
        self.deriv = deriv
        self.__init_params()
        
    def forward(self, x):
        xi1 = self.xi_tilde
        xi2 = torch.transpose(xi1, 1, 2)
        xi3 = torch.transpose(xi2, 0, 2)
        xi4 = torch.transpose(xi3, 1, 2)
        xi5 = torch.transpose(xi4, 0, 2)
        xi6 = torch.transpose(xi5, 1, 2)
        xi = (xi1 - xi2 + xi3 - xi4 + xi5 - xi6) / 6 
        if self.deriv:
            dS = self.S(x)
        else:
            x = x.requires_grad_(True)
            S = self.S(x)
            dS = grad(S, x).reshape([-1,self.ind])
        L = torch.tensordot(dS, xi, dims = ([-1], [-1]))
        return dS, L
        
    def __init_params(self):
        self.xi_tilde = torch.nn.Parameter((torch.randn([self.ind, self.ind, self.ind])*0.1).requires_grad_(True))
        
class gnode_MNN(ln.nn.Module):
    def __init__(self, E, ind, K, deriv = False):
        super(gnode_MNN, self).__init__()
        self.E = E
        self.ind = ind
        self.K = K
        self.deriv = deriv
        self.__init_params()
        
    def forward(self, x):
        lam = (self.lam_tilde - torch.transpose(self.lam_tilde, -1, -2)) / 2
        D = self.D_tilde @ self.D_tilde.t()
        zeta = torch.tensordot(torch.tensordot(lam, D, dims = ([0], [0])), lam, dims = ([2], [0]))
        if self.deriv:
            dE = self.E(x)
        else:
            x = x.requires_grad_(True)
            E = self.E(x)
            dE = grad(E, x).reshape([-1,self.ind])
        dE2 = dE[...,None] @ dE[:,None]
        M = torch.tensordot(dE2, zeta, dims = ([1,2], [1,3]))
        return dE, M.squeeze()
        
    def __init_params(self):
        self.lam_tilde = torch.nn.Parameter((torch.randn([self.K, self.ind, self.ind])*0.1).requires_grad_(True))
        self.D_tilde = torch.nn.Parameter((torch.randn([self.K, self.K])*0.1).requires_grad_(True))
        
class gfinn_LNN(ln.nn.Module):
    def __init__(self, S, ind, K, layers, width, activation, deriv = False):
        super(gfinn_LNN, self).__init__()
        self.S = S
        self.ind = ind
        self.K = K
        self.deriv = deriv
        self.sigComp = ln.nn.FNN(ind, K**2 , layers, width, activation)
        self.__init_params()
        
    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.K, self.K)
        sigma = sigComp - torch.transpose(sigComp, -1, -2)
        if self.deriv:
            dS = self.S(x)
        else:
            x = x.requires_grad_(True)
            S = self.S(x)
            dS = grad(S, x).reshape([-1,self.ind])
        ddS = dS.unsqueeze(-2)
        B = []
        for i in range(self.K):
            xi = torch.triu(self.xi[i], diagonal = 1)
            xi = xi - torch.transpose(xi, -1,-2)
            B.append(ddS@xi)
        B = torch.cat(B, dim = -2)
        L = torch.transpose(B,-1,-2) @ sigma @ B
        return dS, L
        
    def __init_params(self):
        self.xi = torch.nn.Parameter((torch.randn([self.K, self.ind, self.ind])*0.1).requires_grad_(True)) 
        
class gfinn_MNN(ln.nn.Module):
    def __init__(self, E, ind, K, layers, width, activation, deriv = False):
        super(gfinn_MNN, self).__init__()
        self.E = E
        self.ind = ind
        self.K = K
        self.deriv = deriv
        self.sigComp = ln.nn.FNN(ind, K**2 , layers, width, activation)
        self.__init_params()
        
    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.K, self.K)
        sigma = sigComp @ torch.transpose(sigComp, -1, -2)
        if self.deriv:
            dE = self.E(x)
        else:
            x = x.requires_grad_(True)
            E = self.E(x)
            dE = grad(E, x).reshape([-1,self.ind])
        ddE = dE.unsqueeze(-2)
        B = []
        for i in range(self.K):
            xi = torch.triu(self.xi[i], diagonal = 1)
            xi = xi - torch.transpose(xi, -1,-2)
            B.append(ddE@xi)
        B = torch.cat(B, dim = -2)
        M = torch.transpose(B,-1,-2) @ sigma @ B
        return dE, M
        
    def __init_params(self):
        self.xi = torch.nn.Parameter((torch.randn([self.K, self.ind, self.ind])*0.1).requires_grad_(True))
        
class GFINN(ln.nn.LossNN):
    ''' Most useful one: 
    netE: state variable x -> dE, M, orthogonal to each other
    netS: state variable x -> dS, L, orthogonal to each other
    the loss is defined in 'criterion' function
    '''
    def __init__(self, netS, netE, dt, order = 1, iters = 1, lam = 0):
        super(GFINN, self).__init__()
        self.netS = netS
        self.netE = netE
        self.dt = dt
        self.iters = iters
        self.lam = lam
        self.integrator = RK(self.f, order = order, iters = iters)
        self.loss = mse
            
    def f(self, x):
        dE, M = self.netE(x)
        dS, L = self.netS(x)
        dE = dE.unsqueeze(1)
        dS = dS.unsqueeze(1)
        return -(dE @ L).squeeze() + (dS @ M).squeeze() 
    
    def g(self, x):
        return self.netE.B(x)
    
    def consistency_loss(self, x):
        dE, M = self.netE(x)
        dS, L = self.netS(x)
        dEM = dE @ M
        dSL = dS @ L
        return self.lam * (torch.mean(dEM**2) + torch.mean(dSL**2))
        
    def criterion(self, X, y):
        X_next = self.integrator.solve(X, self.dt)
        loss = self.loss(X_next, y) 
        if self.lam > 0:
            loss += self.consistency_loss(X)
        return loss
    
    def predict(self, x0, k, return_np = False):
        x = torch.transpose(self.integrator.flow(x0, self.dt, k - 1), 0, 1)
        if return_np:
            x = x.detach().cpu().numpy()
        return x
