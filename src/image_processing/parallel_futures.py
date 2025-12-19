from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from .filters import grayscale, apply_pipeline
# Import the shared logic
from .utils import to_array, to_image, split_rows, calculate_required_overlap

def _process_chunk_for_executor(arr_chunk, core_slice, steps):
    # Note: imports inside function are sometimes needed for pickling context in some OSs,
    # but generally usually cleaner to have them at top if possible. 
    # For safety with ProcessPoolExecutor, we can keep the imports here if you prefer.
    from .filters import apply_pipeline
    from .utils import to_image # Update this import
    import numpy as np
    
    img = to_image(arr_chunk)
    out = apply_pipeline(img, steps)
    out_arr = np.array(out, dtype=np.float32)
    s, core_start, core_end, e = core_slice
    top = core_start - s
    bottom = top + (core_end - core_start)
    return out_arr[top:bottom, :]

def apply_pipeline_futures(img: Image.Image, steps: list, num_workers: int = 4, overlap: int = None) -> Image.Image:
    if overlap is None:
        overlap = calculate_required_overlap(steps)
        
    g = grayscale(img)
    arr = to_array(g)
    chunks_meta = split_rows(arr, num_workers, overlap=overlap)
    
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