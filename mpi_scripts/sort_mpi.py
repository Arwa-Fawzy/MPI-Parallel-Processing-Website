from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if len(sys.argv) < 2:
        print("Error: No input provided.")
        sys.exit(1)

    input_str = sys.argv[1]  # e.g., "34,7,23"
    data = list(map(int, input_str.strip().split(',')))

    # Pad with large value to preserve sorting correctness
    if len(data) % size != 0:
        pad_len = size - (len(data) % size)
        data += [9999999] * pad_len
else:
    data = None

# Scatter the data
local_data = comm.scatter([data[i::size] for i in range(size)] if rank == 0 else None, root=0)

# Local sort
local_data.sort()

# Gather sorted chunks back to root
gathered = comm.gather(local_data, root=0)

if rank == 0:
    # Flatten and sort the full result
    flat = [item for sublist in gathered for item in sublist]
    flat = [x for x in flat if x != 9999999]  # remove padding
    flat.sort()

    print("Sorted result:", flat)
