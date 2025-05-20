from mpi4py import MPI
import numpy as np
import sys

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define integer matrices
    a = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=np.int32)
    b = np.array([[7, 8],
                  [9, 10],
                  [11, 12]], dtype=np.int32)

    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    if a_cols != b_rows:
        if rank == 0:
            print("Error: incompatible matrix dimensions for multiplication.")
        sys.exit()

    effective_size = min(size, a_rows)

    comm.Bcast(b, root=0)

    rows_per_proc = a_rows // effective_size
    remainder = a_rows % effective_size
    counts = [rows_per_proc + 1 if i < remainder else rows_per_proc for i in range(effective_size)]
    displs = [sum(counts[:i]) for i in range(effective_size)]

    if rank >= effective_size:
        return

    local_rows = counts[rank]
    local_a = np.zeros((local_rows, a_cols), dtype=np.int32)
    if rank == 0:
        for i in range(effective_size):
            start = displs[i]
            end = start + counts[i]
            if i == 0:
                local_a[:] = a[start:end]
            else:
                comm.Send(a[start:end], dest=i, tag=13)
    else:
        comm.Recv(local_a, source=0, tag=13)

    local_c = np.dot(local_a, b)

    if rank == 0:
        c = np.zeros((a_rows, b_cols), dtype=np.int32)
    else:
        c = None

    counts_bytes = [count * b_cols for count in counts]
    displs_bytes = [disp * b_cols for disp in displs]

    comm.Gatherv(local_c, [c, counts_bytes, displs_bytes, MPI.INT], root=0)

    if rank == 0:
        print("Resultant matrix after multiplication:")
        print(c)

if __name__ == "__main__":
    main()
