from mpi4py import MPI
import numpy as np
import sys
import os
from scipy import stats

def get_csv_filename():
    if len(sys.argv) == 2:
        return sys.argv[1]
    else:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if len(csv_files) == 1:
            return csv_files[0]
        elif len(csv_files) > 1:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("Multiple CSV files found, please specify filename.", flush=True)
            return None
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("No CSV file found in current directory. Please specify filename.", flush=True)
            return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = get_csv_filename()
    if filename is None:
        comm.Barrier()
        sys.exit(1)

    data = None
    m = None
    n = None

    if rank == 0:
        try:
            data = np.loadtxt(filename, delimiter=',', skiprows=1)
            m, n = data.shape
        except Exception as e:
            print(f"Error loading CSV file: {e}", flush=True)
            comm.Barrier()
            sys.exit(1)

    m = comm.bcast(m if rank == 0 else None, root=0)
    n = comm.bcast(n if rank == 0 else None, root=0)

    counts = [(m // size) + (1 if i < m % size else 0) for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]

    local_rows = counts[rank]
    local_data = np.zeros((local_rows, n), dtype='d')

    if rank == 0:
        for i in range(size):
            start = displs[i]
            end = start + counts[i]
            if i == 0:
                local_data[:] = data[start:end]
            else:
                comm.Send([data[start:end], MPI.DOUBLE], dest=i, tag=77)
    else:
        comm.Recv([local_data, MPI.DOUBLE], source=0, tag=77)

    local_sum = np.sum(local_data, axis=0)
    local_min = np.min(local_data, axis=0)
    local_max = np.max(local_data, axis=0)
    local_count = local_rows

    total_sum = np.zeros(n, dtype='d')
    comm.Reduce([local_sum, MPI.DOUBLE], [total_sum, MPI.DOUBLE], op=MPI.SUM, root=0)

    total_min = np.zeros(n, dtype='d')
    comm.Reduce([local_min, MPI.DOUBLE], [total_min, MPI.DOUBLE], op=MPI.MIN, root=0)

    total_max = np.zeros(n, dtype='d')
    comm.Reduce([local_max, MPI.DOUBLE], [total_max, MPI.DOUBLE], op=MPI.MAX, root=0)

    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

    gathered_data = None
    if rank == 0:
        gathered_data = np.zeros((m, n), dtype='d')

    comm.Gatherv([local_data, MPI.DOUBLE], [gathered_data, counts, displs, MPI.DOUBLE], root=0)

    if rank == 0:
        mean = total_sum / total_count
        median = np.median(gathered_data, axis=0)
        mode_result = stats.mode(gathered_data, axis=0, keepdims=False)
        mode = mode_result.mode
        std = np.std(gathered_data, axis=0)

        print(f"Statistics for file: {filename}\n")
        for i in range(n):
            print(f"Column {i+1}:")
            print(f" Mean: {mean[i]:.4f}")
            print(f" Median: {median[i]:.4f}")
            print(f" Mode: {mode[i]:.4f}")
            print(f" Min: {total_min[i]:.4f}")
            print(f" Max: {total_max[i]:.4f}")
            print(f" Std Dev: {std[i]:.4f}\n")

if __name__ == "__main__":
    main()
