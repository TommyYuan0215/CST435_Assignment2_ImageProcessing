import numpy as np
from PIL import Image

def to_array(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.float32)

def to_image(arr: np.ndarray) -> Image.Image:
    # Safe casting to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    return Image.fromarray(arr)

def split_rows(arr: np.ndarray, n_chunks: int, overlap: int = 1):
    """Splits an array into horizontal chunks with overlap."""
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
        # Store metadata: (slice_start, core_start, core_end, slice_end)
        chunks.append((s, start, end, e))
        start = end
    return chunks

def calculate_required_overlap(steps: list) -> int:
    """Determines necessary overlap based on filter types."""
    # count how many spatial filters are used
    count = sum(1 for name, _ in steps if name in ('gaussian', 'sobel', 'sharpen'))
    return max(1, count)

def process_chunk_task(args):
    """
    Worker function shared by multiprocessing and futures.
    args: (index, arr_chunk, (s, core_start, core_end, e), steps)
    Returns: (index, processed_array)
    """
    # Import locally to avoid circular imports during module loading,
    # but strictly required for pickle-based multiprocessing.
    from .filters import apply_pipeline_to_array

    index, arr_chunk, (s, core_start, core_end, e), steps = args
    
    # Process the array directly without converting to PIL and back repeatedly
    out_arr = apply_pipeline_to_array(arr_chunk, steps)
    
    # Crop the overlap/halo
    top = core_start - s
    bottom = top + (core_end - core_start)
    
    # Handle both 2D (grayscale) and 3D (color) arrays explicitly
    if out_arr.ndim == 2:
        cropped = out_arr[top:bottom, :]
    elif out_arr.ndim == 3:
        cropped = out_arr[top:bottom, :, :]
    else:
        cropped = out_arr[top:bottom]
    
    return (index, cropped)