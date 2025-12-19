from multiprocessing import Pool
import numpy as np
from PIL import Image
from .filters import grayscale, apply_pipeline
# Import the shared logic
from .utils import to_array, to_image, split_rows, calculate_required_overlap

def _process_chunk(args):
    arr_chunk, core_slice, steps = args
    img = to_image(arr_chunk)
    out = apply_pipeline(img, steps)
    out_arr = np.array(out, dtype=np.float32)
    s, core_start, core_end, e = core_slice
    top = core_start - s
    bottom = top + (core_end - core_start)
    return out_arr[top:bottom, :]

def apply_pipeline_multiprocessing(img: Image.Image, steps: list, num_workers: int = 4, overlap: int = None) -> Image.Image:
    if overlap is None:
        overlap = calculate_required_overlap(steps)
    
    g = grayscale(img)
    arr = to_array(g)
    # Use the shared utility
    chunks_meta = split_rows(arr, num_workers, overlap=overlap)
    
    tasks = []
    for s, core_start, core_end, e in chunks_meta:
        chunk = arr[s:e, :]
        tasks.append((chunk, (s, core_start, core_end, e), steps))

    with Pool(processes=num_workers) as p:
        results = p.map(_process_chunk, tasks)

    out = np.vstack(results)
    return to_image(out)