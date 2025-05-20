from mpi4py import MPI
import sys

def search_keyword(text_chunk, keyword, offset):
    positions = []
    index = text_chunk.find(keyword)
    while index != -1:
        positions.append(offset + index)
        index = text_chunk.find(keyword, index + 1)
    return positions

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) != 3:
        if rank == 0:
            print("Usage: python text_search_mpi.py <text_file> <keyword>")
        sys.exit()

    filename = sys.argv[1]
    keyword = sys.argv[2]

    if rank == 0:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        total_length = len(text)
    else:
        text = None
        total_length = None

    total_length = comm.bcast(total_length, root=0)

    # Calculate chunk sizes
    chunk_size = total_length // size
    start = rank * chunk_size
    # Last process takes the remainder
    end = start + chunk_size if rank != size - 1 else total_length

    # Scatter chunks
    if rank == 0:
        chunks = [text[i*chunk_size : (i+1)*chunk_size] if i != size-1 else text[i*chunk_size:] for i in range(size)]
    else:
        chunks = None

    text_chunk = comm.scatter(chunks, root=0)

    # Each process searches keyword in its chunk
    positions = search_keyword(text_chunk, keyword, start)

    # Gather all positions
    all_positions = comm.gather(positions, root=0)

    if rank == 0:
        # Flatten list of lists
        flat_positions = [pos for sublist in all_positions for pos in sublist]
        print(f"Keyword '{keyword}' found {len(flat_positions)} times at positions:")
        print(flat_positions)

if __name__ == "__main__":
    main()
