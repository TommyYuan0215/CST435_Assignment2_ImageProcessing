from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from .utils import to_array, to_image, split_rows, calculate_required_overlap, process_chunk_task
from .filters import grayscale

def apply_pipeline_futures(img: Image.Image, steps: list, num_workers: int = 4) -> Image.Image:
    # 1. Prepare data
    g = grayscale(to_array(img))
    overlap = calculate_required_overlap(steps)
    chunks_meta = split_rows(g, num_workers, overlap=overlap)
    
    # 2. Prepare tasks
    tasks = [(g[s:e, :], meta, steps) for (s, *_, e), meta in zip(chunks_meta, chunks_meta)]

    # 3. Execute
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_chunk_task, task) for task in tasks]
        results = [f.result() for f in futures]
            
    # 4. Reassemble
    out = np.vstack(results)
    return to_image(out)