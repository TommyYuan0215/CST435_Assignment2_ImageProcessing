"""
Parallel pipeline using the multiprocessing module.
Splits image into horizontal chunks with overlap (1 pixel) for 3x3 kernels.
"""
from multiprocessing import Pool
import numpy as np
from PIL import Image
from .filters import grayscale, apply_pipeline, to_array, to_image


def _split_rows(arr: np.ndarray, n_chunks: int, overlap: int = 1):
    h = arr.shape[0]
    # Ensure n_chunks <= h
    n_chunks = min(n_chunks, h)
    sizes = [h // n_chunks] * n_chunks
    for i in range(h % n_chunks):
        sizes[i] += 1
    chunks = []
    start = 0
    for sz in sizes:
        end = start + sz
        # extend with overlap
        s = max(0, start - overlap)
        e = min(h, end + overlap)
        chunks.append((s, start, end, e))  # (slice_start_incl, core_start, core_end, slice_end_excl)
        start = end
    return chunks


def _process_chunk(args):
    arr_chunk, core_slice, steps = args
    # arr_chunk is 2D grayscale chunk
    img = to_image(arr_chunk)
    out = apply_pipeline(img, steps)
    out_arr = np.array(out, dtype=np.float32)
    # Extract core region
    s, core_start, core_end, e = core_slice
    overlap_top = core_start - s
    top = overlap_top
    bottom = top + (core_end - core_start)
    return out_arr[top:bottom, :]


def _required_overlap_from_steps(steps: list) -> int:
    # Count how many sequential neighborhood (3x3) operations are present
    # Conservative: treat 'gaussian', 'sobel', 'sharpen' as neighborhood ops
    count = 0
    for name, _ in steps:
        if name in ('gaussian', 'sobel', 'sharpen'):
            count += 1
    return max(1, count)


def apply_pipeline_multiprocessing(img: Image.Image, steps: list, num_workers: int = 4, overlap: int = None) -> Image.Image:
    # We'll operate on grayscale
    if overlap is None:
        overlap = _required_overlap_from_steps(steps)
    g = grayscale(img)
    arr = to_array(g)
    chunks_meta = _split_rows(arr, num_workers, overlap=overlap)
    tasks = []
    for s, core_start, core_end, e in chunks_meta:
        chunk = arr[s:e, :]
        tasks.append((chunk, (s, core_start, core_end, e), steps))

    with Pool(processes=num_workers) as p:
        results = p.map(_process_chunk, tasks)

    out = np.vstack(results)
    return to_image(out)
