import numpy as np
from PIL import Image
from .utils import to_array, to_image

# --- Core Filters ---

def grayscale(arr: np.ndarray) -> np.ndarray:
    """Converts (H,W,3) or (H,W,4) array to (H,W) luminance."""
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    # Y = 0.299R + 0.587G + 0.114B
    return np.dot(arr, [0.299, 0.587, 0.114])

def convolve2d_vectorized(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Vectorized convolution for small kernels. 
    Replaces nested loops with array shifting.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    
    # Pad the image
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode='edge')
    h, w = arr.shape
    output = np.zeros_like(arr)
    
    # Iterate over the kernel (e.g., 9 times for 3x3) instead of image (millions of times)
    for i in range(kh):
        for j in range(kw):
            # Shift the padded array to align with kernel position
            # Slice: [i : i + original_height]
            region = padded[i : i + h, j : j + w]
            output += region * kernel[i, j]
            
    return output

def gaussian_blur(arr: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    # Ensure grayscale input for convolution
    if arr.ndim == 3:
        arr = grayscale(arr)
    return convolve2d_vectorized(arr, kernel)

def sobel_edges(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = grayscale(arr)
    
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
    gx = convolve2d_vectorized(arr, kx)
    gy = convolve2d_vectorized(arr, ky)
    
    mag = np.hypot(gx, gy)
    # Normalize
    max_val = mag.max()
    if max_val > 0:
        mag *= 255.0 / max_val
    return mag

def sharpen(arr: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    if arr.ndim == 3:
        arr = grayscale(arr)
    
    # Simple blur for unsharp mask
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    blurred = convolve2d_vectorized(arr, kernel)
    return arr + alpha * (arr - blurred)

def adjust_brightness(arr: np.ndarray, delta: float) -> np.ndarray:
    return arr + delta

# --- Pipeline Logic ---

def apply_pipeline_to_array(arr: np.ndarray, steps: list) -> np.ndarray:
    """Applies steps to a numpy array directly."""
    out = arr.astype(np.float32)
    
    for name, kwargs in steps:
        if name == 'grayscale':
            out = grayscale(out)
        elif name == 'gaussian':
            out = gaussian_blur(out)
        elif name == 'sobel':
            out = sobel_edges(out)
        elif name == 'sharpen':
            out = sharpen(out, **kwargs)
        elif name == 'brightness':
            out = adjust_brightness(out, **kwargs)
        else:
            raise ValueError(f"Unknown step {name}")
    return out

def apply_pipeline(img: Image.Image, steps: list) -> Image.Image:
    """Public interface expecting a PIL Image."""
    arr = to_array(img)
    processed_arr = apply_pipeline_to_array(arr, steps)
    return to_image(processed_arr)