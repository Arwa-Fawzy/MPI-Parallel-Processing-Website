from mpi4py import MPI
import sys
import re

def count_words_parallel(text, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    words = re.findall(r'\b\w+\b', text.lower())
    n = len(words)

    # Divide words among processes
    chunk_size = n // size
    remainder = n % size
    counts = [chunk_size + 1 if i < remainder else chunk_size for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]

    # Scatter words to all processes
    local_words = words[displs[rank]:displs[rank]+counts[rank]]

    # Each process counts local word frequencies
    local_freq = {}
    for w in local_words:
        local_freq[w] = local_freq.get(w, 0) + 1

    # Gather local_freq dicts at root
    all_freqs = comm.gather(local_freq, root=0)

    if rank == 0:
        # Merge all dictionaries
        total_freq = {}
        for freq in all_freqs:
            for k, v in freq.items():
                total_freq[k] = total_freq.get(k, 0) + v

        total_words = sum(total_freq.values())
        unique_words = len(total_freq)

        return total_words, unique_words, total_freq
    else:
        return None, None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: python file_process_mpi.py <input_text_file>")
        sys.exit()

    input_file = sys.argv[1]

    if rank == 0:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = None

    # Broadcast text to all processes
    text = comm.bcast(text, root=0)

    total_words, unique_words, freq_dict = count_words_parallel(text, comm)

    if rank == 0:
        print(f"Total Words: {total_words}")
        print(f"Unique Words: {unique_words}")
        # Optional: print top 10 most frequent words
        top_words = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 words:")
        for word, count in top_words:
            print(f"{word}: {count}")

if __name__ == "__main__":
    main()
