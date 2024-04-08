from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

val = 5*rank

print(f"[Rank, Size, Val]: {rank, size, val}")

reduce_sum = comm.reduce(val, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Summ from reduction: {reduce_sum}")