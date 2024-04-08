from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N=100  # Size of Buffer

send_buf = None

if rank == 0:
    send_buf = np.empty([size, N], dtype='i')
    send_buf.T[:, :] = range(size)

recv_buf = np.empty(N, dtype='i')
comm.Scatter(send_buf, recv_buf, root=0)

print(f"recv_buf: {recv_buf} from rank: {rank}")

assert np.allclose(recv_buf, rank)
