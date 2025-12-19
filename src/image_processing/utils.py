import numpy as np
from PIL import Image

def to_array(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.float32)

def to_image(arr: np.ndarray) -> Image.Image:
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
        chunks.append((s, start, end, e))
        start = end
    return chunks

def calculate_required_overlap(steps: list) -> int:
    """Determines necessary overlap based on filter types."""
    count = 0
    for name, _ in steps:
        # These filters use 3x3 kernels, so they need neighborhood pixels
        if name in ('gaussian', 'sobel', 'sharpen'):
            count += 1
    return max(1, count)