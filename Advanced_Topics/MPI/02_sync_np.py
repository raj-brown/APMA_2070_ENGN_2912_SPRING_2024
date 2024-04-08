from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M = 10
N = 10


# Data Type explicitly
if rank == 0:
    data = np.random.rand(M,N).astype('f')
    for i in range(0, size-1):
        comm.Send([data, MPI.FLOAT], dest=i+1, tag=50+i+1)
else:
    data = np.empty(shape=(M, N), dtype='f')
    comm.Recv([data, MPI.FLOAT], source=0, tag=50 + rank)

print(f"data: {data} from rank: {rank}\n")
MPI.Finalize()

