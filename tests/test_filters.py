import numpy as np
from PIL import Image

from image_processing.filters import apply_pipeline
from image_processing.parallel_multiprocessing import apply_pipeline_multiprocessing
from image_processing.parallel_futures import apply_pipeline_futures


def make_test_image(width=64, height=64):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            arr[i, j] = [(i*255)//(height-1), (j*255)//(width-1), ((i+j)*255)//(2*(width-1))]
    return Image.fromarray(arr)


def test_pipelines_agree():
    img = make_test_image(80, 60)
    steps = [('grayscale', {}), ('gaussian', {}), ('sobel', {}), ('sharpen', {'alpha': 1.0}), ('brightness', {'delta': -10})]

    s = apply_pipeline(img, steps)
    m = apply_pipeline_multiprocessing(img, steps, num_workers=4)
    f = apply_pipeline_futures(img, steps, num_workers=4)

    a = np.array(s, dtype=np.float32)
    b = np.array(m, dtype=np.float32)
    c = np.array(f, dtype=np.float32)

    assert a.shape == b.shape == c.shape
    # Loose tolerance for floating point associativity differences
    assert np.allclose(a, b, atol=1)
    assert np.allclose(a, c, atol=1)
