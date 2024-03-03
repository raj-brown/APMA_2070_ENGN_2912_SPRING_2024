#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 12:54:54 2021

@author: zen
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
        
def plot_GC(data, net):
    def plot_unit(t, y1, y2, ylabel = None):
        plt.plot(t, y1, 'k--', label="True")
        plt.plot(t, y2, 'b', label="Predicted")
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel(r'$t$')
        
    index = 3
    y_true = torch.tensor(data.test_traj, dtype = net.dtype, device = net.device)
    y_true = torch.transpose(y_true, 0, 1)
    y_pred = net.predict(y_true[0], y_true.shape[0])
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    H1_true = y_true[...,1] ** 2 / 2
    H2_true = (np.exp(y_true[...,2]) / y_true[...,0]) ** (2 / 3)
    H3_true = (np.exp(y_true[...,3]) / 2-y_true[...,0]) ** (2 / 3)
    H1_pred = y_pred[...,1] ** 2 / 2
    H2_pred = (np.exp(y_pred[...,2]) / y_pred[...,0]) ** (2 / 3)
    H3_pred = (np.exp(y_pred[...,3]) / 2-y_pred[...,0]) ** (2 / 3)
    scale = np.max(y_true, axis = 0, keepdims = True) - np.min(y_true, axis = 0, keepdims = True)
    rel_error = np.mean(((y_true - y_pred)/scale) ** 2, axis = (1,2))
    t = data.t
    plt.figure(figsize = [4*3,4*2])
    plt.subplot(231)
    plot_unit(t, y_true[:,index,0], y_pred[:,index,0], r'$q$')
    plt.subplot(232)
    plot_unit(t, y_true[:,index,1], y_pred[:,index,1], r'$v$')
    plt.subplot(233)
    plot_unit(t, y_true[:,index,2], y_pred[:,index,2], r'$s_1$')
    plt.subplot(234)
    plot_unit(t, y_true[:,index,3], y_pred[:,index,3], r'$s_2$')
    plt.subplot(235)
    plot_unit(t, H1_true[:,index], H1_pred[:,index])
    plot_unit(t, H2_true[:,index], H2_pred[:,index])
    plot_unit(t, H3_true[:,index], H3_pred[:,index], 'Energy')
    plt.subplot(236)
    plt.plot(t, rel_error, label = 'Normalized Prediction Error')
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel(r'$t$')
    plt.tight_layout()
    if not os.path.exists('figs'): os.mkdir('figs')
    plt.savefig('figs/gc_{}_{}.pdf'.format('gfinn2b', 3))
