from multiprocessing import Pool
import numpy as np
from PIL import Image
from .utils import to_array, to_image, split_rows, calculate_required_overlap, process_chunk_task

# --- GLOBAL VARIABLES (The "Memory") ---
_POOL = None
_CURRENT_WORKERS = 0

def apply_pipeline_multiprocessing(img: Image.Image, steps: list, num_workers: int = 4) -> Image.Image:
    global _POOL, _CURRENT_WORKERS

    # 1. Manage the Global Pool (Outer Pool Logic)
    if _POOL is None or _CURRENT_WORKERS != num_workers:
        if _POOL is not None:
            _POOL.close()
            _POOL.join()
        _POOL = Pool(processes=num_workers)
        _CURRENT_WORKERS = num_workers

    # 2. Prepare data
    arr = to_array(img)
    overlap = calculate_required_overlap(steps)
    chunks_meta = split_rows(arr, num_workers, overlap=overlap)
    
    # 3. Prepare tasks
    tasks = [(arr[s:e, :], meta, steps) for (s, *_, e), meta in zip(chunks_meta, chunks_meta)]

    # 4. Execute using the CACHED global pool
    # This is fast because _POOL is reused!
    results = _POOL.map(process_chunk_task, tasks)

    # 5. Reassemble
    out = np.vstack(results)
    return to_image(out)