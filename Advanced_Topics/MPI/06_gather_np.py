from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N=100  # Size of Buffer
send_buf = np.zeros(N, dtype='i') + rank
recv_buf = None

if rank == 0:
    recv_buf = np.empty([size, N], dtype='i')

comm.Gather(send_buf, recv_buf, root=0)

if rank==0:
    for i in range(size):
        assert np.allclose(recv_buf[i, :], i)

print(f"Test passed from rank: {rank}")
