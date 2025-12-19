"""
Image processing filters: grayscale, gaussian blur (3x3), sobel, sharpen, brightness
"""
from PIL import Image
import numpy as np

# Utility helpers

def to_array(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.float32)


def to_image(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    return Image.fromarray(arr)


# 1. Grayscale using luminance formula
def grayscale(img: Image.Image) -> Image.Image:
    a = to_array(img)
    if a.ndim == 2:
        return img.convert('L')
    # If RGBA, drop alpha
    if a.shape[2] == 4:
        a = a[:, :, :3]
    # Luminance Y = 0.299 R + 0.587 G + 0.114 B
    y = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return to_image(y)


# Convolution helper with small kernels
def convolve2d_gray(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # arr: 2D grayscale float32
    h, w = arr.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    out = np.zeros_like(arr)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    return out


# 2. Gaussian blur 3x3
def gaussian_blur(img: Image.Image) -> Image.Image:
    gray = grayscale(img)
    a = to_array(gray)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    blurred = convolve2d_gray(a, kernel)
    return to_image(blurred)


# 3. Sobel edge detection
def sobel_edges(img: Image.Image) -> Image.Image:
    gray = grayscale(img)
    a = to_array(gray)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve2d_gray(a, kx)
    gy = convolve2d_gray(a, ky)
    mag = np.hypot(gx, gy)
    mag *= 255.0 / (mag.max() + 1e-6)
    return to_image(mag)


# 4. Sharpening - unsharp mask like: result = img + alpha * (img - blurred)
def sharpen(img: Image.Image, alpha: float = 1.0) -> Image.Image:
    gray = grayscale(img)
    a = to_array(gray)
    # simple 3x3 blur kernel
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    blurred = convolve2d_gray(a, kernel)
    result = a + alpha * (a - blurred)
    return to_image(result)


# 5. Brightness adjustment: delta in [-255, 255]
def adjust_brightness(img: Image.Image, delta: float) -> Image.Image:
    a = to_array(img)
    if a.ndim == 2:
        out = a + delta
        return to_image(out)
    out = a + delta
    return to_image(out)


# Small serial pipeline utility
def apply_pipeline(img: Image.Image, steps: list) -> Image.Image:
    """steps: list of tuples (name, kwargs dict)
    Supported names: 'grayscale','gaussian','sobel','sharpen','brightness'
    """
    out = img
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
