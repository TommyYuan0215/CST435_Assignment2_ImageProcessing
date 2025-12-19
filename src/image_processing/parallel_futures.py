"""
Parallel pipeline using concurrent.futures (ProcessPoolExecutor).
"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from .filters import grayscale, apply_pipeline, to_array, to_image

from typing import List


def _split_rows(arr: np.ndarray, n_chunks: int, overlap: int = 1):
    h = arr.shape[0]
    n_chunks = min(n_chunks, h)
    sizes = [h // n_chunks] * n_chunks
    for i in range(h % n_chunks):
        sizes[i] += 1
    chunks = []
    start = 0
    for sz in sizes:
        end = start + sz
        s = max(0, start - overlap)
        e = min(h, end + overlap)
        chunks.append((s, start, end, e))
        start = end
    return chunks


def _process_chunk_for_executor(arr_chunk: np.ndarray, core_slice, steps: List):
    from .filters import apply_pipeline, to_image  # local import for worker process
    import numpy as np
    from PIL import Image
    img = to_image(arr_chunk)
    out = apply_pipeline(img, steps)
    out_arr = np.array(out, dtype=np.float32)
    s, core_start, core_end, e = core_slice
    overlap_top = core_start - s
    top = overlap_top
    bottom = top + (core_end - core_start)
    return out_arr[top:bottom, :]


def _required_overlap_from_steps(steps: list) -> int:
    count = 0
    for name, _ in steps:
        if name in ('gaussian', 'sobel', 'sharpen'):
            count += 1
    return max(1, count)


def apply_pipeline_futures(img: Image.Image, steps: list, num_workers: int = 4, overlap: int = None) -> Image.Image:
    if overlap is None:
        overlap = _required_overlap_from_steps(steps)
    g = grayscale(img)
    arr = to_array(g)
    chunks_meta = _split_rows(arr, num_workers, overlap=overlap)
    tasks = []
    for s, core_start, core_end, e in chunks_meta:
        chunk = arr[s:e, :]
        tasks.append((chunk, (s, core_start, core_end, e), steps))

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_process_chunk_for_executor, chunk, core_slice, steps)
                   for (chunk, core_slice, _steps) in tasks]
        for f in futures:
            results.append(f.result())
    out = np.vstack(results)
    return to_image(out)
