import os
import tensorflow as tf
from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

#np.random.seed(1234)
#tf.random.set_seed(1234)

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

@tf.function
def pde_nn(X, W, b):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(X)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(X)
            u = DNN(X, W, b)
        u_x = tape1.gradient(u, X)
       #del tape1
    u_xx = tape2.gradient(u_x, X)
    #del tape2
    f = 4*tf.sin(2*np.pi*X)*np.pi*np.pi
    R = u_xx + f
    return R
    

@tf.function
def flux_nn(X, W, b):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        u = DNN(X, W, b)
    u_x = tape.gradient(u, X)
    del tape
    return -u_x, u


def trainable_variables(W, b):
    return W + b




    

if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')

    print ('This is from rank %d'%(rank))
    
    if rank == 0:
        N = 10 # Number of Residual trainning data
        x_f = np.linspace(-1, 0.0, N).reshape((-1, 1))        
    else:
        N = 10 # Number of Residual trainning data
        x_f = np.linspace(0.0, 1.0, N).reshape((-1, 1))
        
    if rank == 0:
        x = np.array([-1]).reshape((-1, 1))
    
    if rank == 1:
        x = np.array([1]).reshape((-1, 1))
    
    y = np.sin(2*np.pi*x)    
    x_int = np.array([0.0]).reshape((-1, 1))
    x_f =  tf.convert_to_tensor(x_f, dtype=tf.float32)
    x_train = tf.convert_to_tensor(x, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y, dtype=tf.float32)
    x_int =  tf.convert_to_tensor(x_int, dtype=tf.float32)




    layers = [1] + 2*[16] + [1]
    L = len(layers)
    W = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
    b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)]     
    lr = 0.001
    opt = tf.optimizers.Adam(lr)    
    Nmax = 15000 # Iteration counter

    def loss_fn(x, y, x_f, xint, W, b):
        y_nn = DNN(x, W, b)
        R_nn = pde_nn(x_f, W, b)    
        u_x, u = flux_nn(xint, W, b)
        u_np = u.numpy()
        u_x_np = u_x.numpy()
        u_prev = np.zeros_like(u_np)
        u_next = np.zeros_like(u_np)
        u_x_prev =  np.zeros_like(u_np)
        u_x_next =  np.zeros_like(u_np)
        if rank == 0:
            comm.Send(u_np, dest=1, tag=11)
            comm.Recv(u_next, source=1, tag=12)
            comm.Send(u_x_np, dest=1, tag=13)
            comm.Recv(u_x_next, source=1, tag=14)
            u_next_tf = tf.convert_to_tensor(u_next)
            u_x_next_tf = tf.convert_to_tensor(u_x_next)
            loss_u = u - 0.5*(u + u_next)
            loss_f = u_x - 0.5*(u_x + u_x_next)
            loss_uf = tf.reduce_mean(tf.square(loss_u)) + tf.reduce_mean(tf.square(loss_f))
        elif rank == 1:
            comm.Recv(u_prev, source=0, tag=11)
            comm.Send(u_np,  dest=0, tag=12)
            comm.Recv(u_x_prev, source=0, tag=13)
            comm.Send(u_x_np,  dest=0, tag=14)
            loss_u = u - 0.5*(u + u_prev)
            loss_f = u_x - 0.5*(u_x + u_x_prev)
            loss_uf = tf.reduce_mean(tf.square(loss_u)) + tf.reduce_mean(tf.square(loss_f))
        #sys.exit()
        comm.Barrier()
        loss = tf.reduce_mean(tf.square(y_nn - y)) + tf.reduce_mean(tf.square(R_nn)) + loss_uf
        return loss, loss_uf, y_nn

    #@tf.function
    def training_step(x, y, x_f, W, b, opt, xint):
        with tf.GradientTape() as tape:
            tape.watch([W, b])
            loss, loss_uf, y_nn = loss_fn(x, y, x_f, xint, W, b)
        grads = tape.gradient(loss, trainable_variables(W, b))
        opt.apply_gradients(zip(grads, trainable_variables(W, b)))
        return loss, loss_uf, y_nn

    start_time = time.perf_counter()
    
    for n in range(0, Nmax):
        loss_, loss_int_, y_pred = training_step(x_train, y_train, x_f,  W, b, opt, x_int)

        if n%100 == 0:
            print('rank %d: n = %d, loss = %.3e, loss_int=%.3e'%(rank, n, loss_, loss_int_))
    stop_time = time.perf_counter()
    print('Rank: %d, Elapsed time: %f s'%(rank, stop_time - start_time))
     
    if rank == 0:
        N_plot = 101
        xplot = np.linspace(-1, 0, N_plot).reshape((-1, 1))
        y_col =  np.sin(2*np.pi*xplot) 
        xplot_tf = tf.convert_to_tensor(xplot, dtype=tf.float32) 
        y_pred_ = DNN(xplot_tf, W, b)
        filename_x = 'x_pred_' +  str(rank)
        filename_y = 'y_pred_' + str(rank)
        filename_y_act = 'y_act_' + str(rank)     
        np.savetxt(filename_x, xplot, fmt='%e')
        np.savetxt(filename_y, y_pred_, fmt='%e')
        np.savetxt(filename_y_act, y_col, fmt='%e')
    else:
        N_plot = 101
        xplot = np.linspace(0, 1, N_plot).reshape((-1, 1)) 
        y_col = np.sin(2*np.pi*xplot) 
        xplot_tf = tf.convert_to_tensor(xplot, dtype=tf.float32) 
        y_pred_ = DNN(xplot_tf, W, b)
        filename_x = 'x_pred_' +  str(rank)
        filename_y =  'y_pred_' + str(rank)
        filename_y_act = 'y_act_' + str(rank)     
        np.savetxt(filename_y, y_pred_, fmt='%e')
        np.savetxt(filename_x, xplot, fmt='%e')
        np.savetxt(filename_y_act, y_col, fmt='%e')
