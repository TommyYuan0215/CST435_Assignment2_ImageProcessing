from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from .utils import to_array, to_image, split_rows, calculate_required_overlap, process_chunk_task

# --- GLOBAL VARIABLES (The "Memory") ---
_EXECUTOR = None
_CURRENT_WORKERS = 0

def apply_pipeline_futures(img: Image.Image, steps: list, num_workers: int = 4) -> Image.Image:
    global _EXECUTOR, _CURRENT_WORKERS

    # 1. Manage the Global Executor (Outer Pool Logic)
    # If executor doesn't exist or worker count changed, rebuild it.
    if _EXECUTOR is None or _CURRENT_WORKERS != num_workers:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=True)
        # Using ThreadPoolExecutor as decided for GCP performance
        _EXECUTOR = ThreadPoolExecutor(max_workers=num_workers)
        _CURRENT_WORKERS = num_workers

    # 2. Prepare data
    arr = to_array(img)
    overlap = calculate_required_overlap(steps)
    chunks_meta = split_rows(arr, num_workers, overlap=overlap)
    
    # 3. Prepare tasks with index and array copies for thread safety
    tasks = [
        (i, arr[s:e, :].copy(), meta, steps) 
        for i, ((s, *_, e), meta) in enumerate(zip(chunks_meta, chunks_meta))
    ]

    # 4. Execute using the CACHED global executor
    futures = [_EXECUTOR.submit(process_chunk_task, task) for task in tasks]
    results_with_index = [f.result() for f in futures]
    
    # Sort by index to ensure deterministic ordering
    results_with_index.sort(key=lambda x: x[0])
    results = [r[1] for r in results_with_index]
            
    # 5. Reassemble
    out = np.vstack(results)
    return to_image(out)