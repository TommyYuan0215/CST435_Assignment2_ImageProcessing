"""
CLI to run pipelines and compare results
Usage: python run_pipeline.py input.jpg out_prefix --workers 4
"""
import argparse
import time
from PIL import Image
from src.image_processing.filters import apply_pipeline
from src.image_processing.parallel_multiprocessing import apply_pipeline_multiprocessing
from src.image_processing.parallel_futures import apply_pipeline_futures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image path')
    parser.add_argument('out_prefix', help='Output file prefix')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    img = Image.open(args.input).convert('RGB')
    steps = [('grayscale', {}), ('gaussian', {}), ('sobel', {}), ('sharpen', {'alpha': 1.0})]

    t0 = time.time()
    out_serial = apply_pipeline(img, steps)
    ts = time.time() - t0
    out_serial.save(f"{args.out_prefix}_serial.png")
    print(f"Serial pipeline: {ts:.3f}s -> {args.out_prefix}_serial.png")

    t0 = time.time()
    out_mp = apply_pipeline_multiprocessing(img, steps, num_workers=args.workers)
    tmp = time.time() - t0
    out_mp.save(f"{args.out_prefix}_mp.png")
    print(f"multiprocessing: {tmp:.3f}s -> {args.out_prefix}_mp.png")

    t0 = time.time()
    out_fut = apply_pipeline_futures(img, steps, num_workers=args.workers)
    tf = time.time() - t0
    out_fut.save(f"{args.out_prefix}_futures.png")
    print(f"concurrent.futures: {tf:.3f}s -> {args.out_prefix}_futures.png")

    # Quick check: images are equal-ish
    import numpy as np
    s = np.array(out_serial, dtype=np.int32)
    m = np.array(out_mp, dtype=np.int32)
    f = np.array(out_fut, dtype=np.int32)
    print('MP equal to serial?', np.allclose(s, m))
    print('Futures equal to serial?', np.allclose(s, f))

if __name__ == '__main__':
    main()
