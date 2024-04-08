from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Printing from Rank: {rank} from the {size} pool of Processors.")
    data = {"Course": "APMA-2070", "School": "Brown University"}
    req = comm.isend(data, dest=1, tag=10)
    req.wait()
else:
    data = None
    print(f"Data: {data} and Rank:{rank}")
    req = comm.irecv(source=0, tag=10)
    z = 2*2
    

    data = req.wait()
    print(f"Data Recieved Asynchronously on Rank: {rank} and data is: {data} ")

MPI.Finalize()


