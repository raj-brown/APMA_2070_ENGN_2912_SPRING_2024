import os
import tensorflow as tf
from mpi4py import MPI
import horovod.tensorflow as hvd
import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(1234)
tf.random.set_seed(1234)

def hyper_initial(size):
    in_dim = size[0]
    out_dim = size[1]
    std = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal(shape=size, stddev = std))

def DNN(X, W, b):
    A = X
    L = len(W)
    for i in range(L-1):
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y



def trainable_variables(W, b):
    return W + b


@tf.function
def training_step(x, y, W, b, opt, it):
    with tf.GradientTape() as tape:
        tape.watch([W, b])
        y_nn = DNN(x, W, b) 
        loss = tf.reduce_mean(tf.square(y_nn - y)) 

    #tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss, trainable_variables(W, b))
    opt.apply_gradients(zip(grads, trainable_variables(W, b)))

    if it == True:
        hvd.broadcast_variables(trainable_variables(W, b), root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return loss, y_nn

if __name__ == '__main__':
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    print ('This is from rank %d'%(hvd.rank()))
    if hvd.rank() == 0:
        N = 16 # Number of trainning data
        x_col = np.linspace(-1, 0, N).reshape((-1, 1))
    else:
        N = 16 # Number of trainning data
        x_col = np.linspace(0, 1, N).reshape((-1, 1))

    #print('x_col from rank %d'%(hvd.rank()))
    #print (x_col)

    layers = [1] + 2*[16] + [1]
    L = len(layers)
    W = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
    b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)]     
    
    lr = 0.001*hvd.size()
    opt = tf.optimizers.Adam(0.001 * hvd.size())    
    if hvd.rank() == 0:
        y_col = np.sin(2*np.pi*x_col) + np.sin(4*np.pi*x_col)
    else:
        y_col = np.sin(2*np.pi*x_col) + np.sin(4*np.pi*x_col)

    x_train = tf.convert_to_tensor(x_col, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_col, dtype=tf.float32)

    Nmax = 30000 # Iteration counter
    start_time = time.perf_counter()
    
    for n in range(0, Nmax):
        if n==0:
            loss_, y_pred = training_step(x_train, y_train, W, b, opt, True)
        else:
            loss_, y_pred = training_step(x_train, y_train, W, b, opt, False)


        if n%100 == 0:
            print('rank %d: n = %d, loss = %.3e'%(hvd.rank(), n, loss_))
    stop_time = time.perf_counter()
    print('Rank: %d, Elapsed time: %f s'%(hvd.rank(), stop_time - start_time))
     
    if hvd.rank() == 0:
        N_plot = 101
        xplot = np.linspace(-1, 1, N_plot).reshape((-1, 1))
        y_col = np.sin(2*np.pi*xplot) + np.sin(4*np.pi*xplot)
        xplot_tf = tf.convert_to_tensor(xplot, dtype=tf.float32) 
        y_pred_ = DNN(xplot_tf, W, b)
        filename_x = 'x_pred_' +  str(hvd.rank())
        filename_y = 'y_pred_' + str(hvd.rank())
        filename_y_act = 'y_act_' + str(hvd.rank())     
        np.savetxt(filename_x, xplot, fmt='%e')
        np.savetxt(filename_y, y_pred_, fmt='%e')
        np.savetxt(filename_y_act, y_col, fmt='%e')
    else:
        N_plot = 101
        xplot = np.linspace(-1, 1, N_plot).reshape((-1, 1)) 
        y_col = np.sin(2*np.pi*xplot) + np.sin(4*np.pi*xplot)
        xplot_tf = tf.convert_to_tensor(xplot, dtype=tf.float32) 
        y_pred_ = DNN(xplot_tf, W, b)
        filename_x = 'x_pred_' +  str(hvd.rank())
        filename_y =  'y_pred_' + str(hvd.rank())
        filename_y_act = 'y_act_' + str(hvd.rank())     
        np.savetxt(filename_y, y_pred_, fmt='%e')
        np.savetxt(filename_x, xplot, fmt='%e')
        np.savetxt(filename_y_act, y_col, fmt='%e')
