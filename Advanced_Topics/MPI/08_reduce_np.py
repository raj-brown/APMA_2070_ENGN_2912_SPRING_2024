from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = np.zeros(4)
for i in range(comm.rank, len(data), comm.size):
    data[i] = i

print(f"Data: {data} and Rank: {rank}")


if rank==0:
    reduce_array = np.zeros_like(data)
else:
    reduce_array=None


comm.Reduce([data, MPI.INT],  [reduce_array, MPI.INT], op=MPI.SUM, root=0)

print(f"Rank: {rank}, Reduce_array: {reduce_array}")

