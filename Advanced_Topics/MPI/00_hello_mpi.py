from mpi4py import MPI
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


print(f"Non-ordered hello from rank: {rank} from the world of: {size} processors!\n")

sys.exit()
if (rank == 0):
    print("**************+++++**************\n")

s = 0
while s < size:
    if rank == s:
        print(f"Ordered Hello from rank: {rank} from the world of: {size} processors!\n")
    s = s + 1
    comm.Barrier()


