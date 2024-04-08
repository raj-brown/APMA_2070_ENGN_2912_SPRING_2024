from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N=100  # Size of Buffer

if rank == size-1:
    data = np.arange(N, dtype='i')
else:
    data = np.empty(N, dtype='i')

comm.Bcast(data, root=size-1)

for i in range(0, N):
    print(f"Rank: {rank}")
    assert data[i] == i 

