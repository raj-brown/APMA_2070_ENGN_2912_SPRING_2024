from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    N = 100
else:
    N = None

N = comm.bcast(N, root=0)
print(f"Rank: {rank} and N: {N}")
h = 1.0 / N; 
s = 0.0

for i in range(rank, N, size):
    x = h * (i + 0.5)
    s += 4.0 / (1.0 + x**2)

PI = np.array(s * h, dtype='d')

PI_TOTAL = np.zeros_like(PI)

comm.Reduce([PI, MPI.DOUBLE], [PI_TOTAL, MPI.DOUBLE],
            op=MPI.SUM, root=0)
print(f"Rank: {rank} and PI: {PI_TOTAL}")
