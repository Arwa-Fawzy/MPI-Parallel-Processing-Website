# mpi_scripts/image_process_mpi.py

from mpi4py import MPI
import cv2
import numpy as np
import sys
import os

def apply_filter(image_chunk, filter_type):
    if filter_type == 'grayscale':
        return cv2.cvtColor(image_chunk, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        return cv2.GaussianBlur(image_chunk, (5, 5), 0)
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

def split_image(image, num_chunks):
    """Splits image into vertical stripes."""
    height = image.shape[0]
    chunk_heights = np.linspace(0, height, num_chunks + 1, dtype=int)
    return [image[chunk_heights[i]:chunk_heights[i+1]] for i in range(num_chunks)]

def stitch_chunks(chunks):
    return np.vstack(chunks)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if len(sys.argv) != 3:
            print("Usage: python image_process_mpi.py <image_path> <filter_type>")
            sys.exit(1)

        image_path = sys.argv[1]
        filter_type = sys.argv[2]

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            sys.exit(1)

        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image")
            sys.exit(1)

        # Split image vertically
        image_chunks = split_image(image, size)
    else:
        image_chunks = None
        filter_type = None

    # Broadcast filter type to all processes
    filter_type = comm.bcast(filter_type if rank == 0 else None, root=0)

    # Scatter image chunks
    image_chunk = comm.scatter(image_chunks, root=0)

    # Each process applies the filter to its chunk
    processed_chunk = apply_filter(image_chunk, filter_type)

    # Gather the processed chunks
    gathered_chunks = comm.gather(processed_chunk, root=0)

    if rank == 0:
        result_image = stitch_chunks(gathered_chunks)

        # Generate output file name
        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        output_filename = f"results/{name}_{filter_type}{ext}"
        os.makedirs("results", exist_ok=True)

        # Save the final processed image
        if filter_type == "grayscale":
            cv2.imwrite(output_filename, result_image)
        else:
            cv2.imwrite(output_filename, result_image)

        print(f"Image processed and saved as: {output_filename}")

if __name__ == "__main__":
    main()
