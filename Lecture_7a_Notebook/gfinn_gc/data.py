#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:36:11 2021

@author: zen
"""
import learner_gfinn as ln
import numpy as np
import os
from learner_gfinn.integrator import RK # runge kutta for deterministic system

class Data(ln.Data):
    def __init__(self, split_ratio, problem, t_terminal, dt, trajs, order, iters, sigma, init, noise = 0, new = False):
        super(Data, self).__init__()
        if problem == 'GC': # gas container
            self.integrator = RK(self.__gas_container, order=order, iters=iters)
        else:
            raise NotImplementedError
            
        # bolzmann constant for stochastic system, 0 for determininstic ones
        self.sigma = sigma
        self.init = init
        
        # Generate several trajectories for training and testing
        data = self.__generate_flow(problem = problem, file = problem, t_terminal = t_terminal, dt = dt, trajs = trajs, new = new)
        self.dt = data['dt']
        self.t = data['t_vec']
        self.num_t = len(self.t)
        self.dims = data['Z'].shape[-1]
        self.noise = noise
        
        # From the trajectories data, we do the train-test split and group them into
        # data pairs for future training
        self.__generate_data(data['Z'], split_ratio)
    
    def __generate_data(self, X, split):
        # train-test split
        num_train = int(len(X)*split)
        self.train_traj = X[:num_train]
        self.test_traj = X[num_train:]
        std = np.std(self.train_traj, (0,1), keepdims = True)
        
        # add noise to the data
        self.train_traj += self.noise * np.random.randn(*self.train_traj.shape) * std
        
        # group the trajectories data into input-target pairs, that can be fed directly into NNs
        X_train, y_train = self.train_traj[:,:-1], self.train_traj[:,1:]
        X_test, y_test = self.test_traj[:,:-1], self.test_traj[:,1:]
        self.X_train = X_train.reshape([-1,self.dims])
        self.X_test = X_test.reshape([-1,self.dims])
        self.y_train = y_train.reshape([-1,self.dims])
        self.y_test = y_test.reshape([-1,self.dims])
    
    def __gas_container(self, x):
        alpha, c = 10, 1
        q, p, S1, S2 = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:]
        E1 = (np.exp(S1) / q / c) ** (2 / 3)
        E2 = (np.exp(S2) / (2 - q) / c) ** (2 / 3)
        q_dot = p
        p_dot = 2 / 3 * (E1 / q - E2 / (2 - q))
        S1_dot = 9 * alpha / 4 / E1 * (1 / E1 - 1 / E2)
        S2_dot = - 9 * alpha / 4 / E2 * (1 / E1 - 1 / E2)
        return np.concatenate([q_dot, p_dot, S1_dot, S2_dot], axis = -1)
    
    def __generate_flow(self, file, new, trajs = 3, t_terminal = 5, dt = 0.05, problem = 'GC'):
        data = {}
        path = 'data/database_{}.npy'.format(file)
        if os.path.exists(path) and (not new):
            data = np.load(path, allow_pickle=True).item()
            return data

        t = np.linspace(0, t_terminal, int(t_terminal / dt) + 1)
        
        # specify the initial conditions
        if problem == 'GC':
            x0 = np.array([self.init]) + (2*np.random.rand(trajs, 4) - 1) * np.array([[0.8,1,1,1]]) * self.sigma
        
        # solve the ODE using predefined numerical integrators
        Z = self.integrator.flow(x0, dt, int(t_terminal / dt))
        data['Z'] = Z
        data['dt'] = dt
        data['t_vec'] = t
        if not os.path.exists('data'): os.mkdir('data')
        if not os.path.exists(path): np.save(path, data)
        return data
        
def test_gc():
    np.random.seed(2)   
    p = 0.8
    problem = 'GC'
    t_terminal = 8
    dt = 0.02
    trajs = 100
    order = 2
    iters = 1
    sigma = 1
    init = [1,0,2,2]
    data = Data(p, problem, t_terminal, dt, trajs, order, iters, sigma = sigma, init = init, new = True, noise = 0)
    import matplotlib.pyplot as plt
    print(data.train_traj.shape)
    x = data.train_traj[2]
    plt.plot(data.t, x[:,0], label = 'q')
    plt.plot(data.t, x[:,1], label = 'p')
    plt.plot(data.t, x[:,2], label = '$S_1$')
    plt.plot(data.t, x[:,3], label = '$S_2$')
    plt.legend()
    plt.tight_layout()
    plt.xlabel('t')
    plt.ylabel('One sample path')
    if not os.path.exists('figs'): os.mkdir('figs')
    plt.savefig('figs/gc_sample.pdf')
    plt.show()
    plt.figure(figsize=[4,4])
    plt.plot(x[:,0],x[:,1])
  
if __name__ == '__main__':
    test_gc()
    
        
