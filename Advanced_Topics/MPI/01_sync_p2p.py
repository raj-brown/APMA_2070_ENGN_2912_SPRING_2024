from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Printing from Rank: {rank} from the {size} pool of Processors.")
    data = {"Course": "APMA-2070", "School": "Brown University"}
    comm.send(data, dest=1, tag=10)
else:
    data = None
    print(f"Data: {data} and Rank: {rank}")
    data = comm.recv(source=0, tag=10)
    print(f"Data Recieved on Rank: {rank} and data is: {data} ")


MPI.Finalize()


