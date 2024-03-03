#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:59:32 2021

gas container example

@author: zen
"""

import learner as ln
import torch
import numpy as np
import argparse

from data import Data
from nn import *
from postprocess_gc import plot_GC
        
def main(args):
    device = 'cpu' # 'cpu' or 'gpu'
    dtype = 'float'
    # data
    p = 0.8
    problem = 'GC'
    t_terminal = 8
    dt = 0.02
    iters = 1
    trajs = 100
    order = 2
    sigma = 1
    init = [1,0,2,2]
    
    data = Data(p, problem, t_terminal, dt, trajs, order, iters, sigma = sigma, init=init, new = True, noise = 0)
    # NN
    ind = 4
    layers = 5
    width  = 30
    activation = 'tanh'

    batch_size = 100
    lr = 0.001

    if args.net == 'gfinn2a':
        S = GC_S()
        E = GC_E()
        netS = gfinn_LNN(S, ind = ind, K = ind, layers = layers, width = width, activation = activation, deriv = True)
        netE = gfinn_MNN(E, ind = ind, K = ind, layers = layers, width = width, activation = activation, deriv = True)
        net = GFINN(netS, netE, data.dt / iters, order = order, iters = iters)
    elif args.net == 'gfinn2b':
        S = ln.nn.FNN(ind, 1, layers=1, width=width, activation=activation)
        E = ln.nn.FNN(ind, 1, layers=layers, width=width, activation=activation)
        netS = gfinn_LNN(S, ind = ind, K = ind, layers = layers, width = width, activation = activation)
        netE = gfinn_MNN(E, ind = ind, K = ind, layers = layers, width = width, activation = activation)
        net = GFINN(netS, netE, data.dt / iters, order = order, iters = iters)
    elif args.net == 'gnode2a':
        S = GC_S()
        E = GC_E()
        netS = gnode_LNN(S, ind = ind, deriv = True)
        netE = gnode_MNN(E, ind = ind, K = ind, deriv = True)
        net = GFINN(netS, netE, data.dt / iters, order = order, iters = iters)
    elif args.net == 'gnode2b':
        S = ln.nn.FNN(ind, 1, layers=1, width=width, activation=activation)
        E = ln.nn.FNN(ind, 1, layers=layers, width=width, activation=activation)
        netS = gnode_LNN(S, ind = ind)
        netE = gnode_MNN(E, ind = ind, K = ind)
        net = GFINN(netS, netE, data.dt / iters, order = order, iters = iters)
    else:
        raise NotImplementedError
    
    # training
    iterations  = 100000
    lbfgs_steps = 0
    print_every = 100
    path = problem + args.net + str(args.lam) + '_' + str(args.seed) 
    callback = None
    
    args2 = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'lbfgs_steps': lbfgs_steps,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'path': path,
        'callback': callback,
        'dtype': dtype,
        'device': device,
    }
    
    ln.Brain.Init(**args2)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    plot_GC(data, net, args)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Neural Networks')
    parser.add_argument('--net', default='gfinn2b', type=str, help='gfinn2a or gfinn2b or gnode2a or gnode2b')
    parser.add_argument('--seed', default=3, type=int, help='random seed')
    parser.add_argument('--lam', default=0, type=float, help='lambda as the weight for consistency penalty')
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)
