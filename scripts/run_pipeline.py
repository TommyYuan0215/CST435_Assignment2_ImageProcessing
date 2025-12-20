"""
CLI to run pipelines and compare results
Usage: python -m scripts.run_pipeline input.jpg out_prefix --workers 4
"""
import argparse
import time
import numpy as np
from PIL import Image

# UPDATED IMPORTS: Removed 'src.' prefix
from image_processing.filters import apply_pipeline
from image_processing.parallel_multiprocessing import apply_pipeline_multiprocessing
from image_processing.parallel_futures import apply_pipeline_futures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image path')
    parser.add_argument('out_prefix', help='Output file prefix')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    img = Image.open(args.input).convert('RGB')
    steps = [('grayscale', {}), ('gaussian', {}), ('sobel', {}), ('sharpen', {'alpha': 1.0})]

    print("Running Serial...")
    t0 = time.time()
    out_serial = apply_pipeline(img, steps)
    print(f"Serial: {time.time() - t0:.3f}s")
    out_serial.save(f"{args.out_prefix}_serial.png")

    print(f"Running Multiprocessing ({args.workers} workers)...")
    t0 = time.time()
    out_mp = apply_pipeline_multiprocessing(img, steps, num_workers=args.workers)
    print(f"Multiprocessing: {time.time() - t0:.3f}s")
    out_mp.save(f"{args.out_prefix}_mp.png")

    print(f"Running Futures ({args.workers} workers)...")
    t0 = time.time()
    out_fut = apply_pipeline_futures(img, steps, num_workers=args.workers)
    print(f"Futures: {time.time() - t0:.3f}s")
    out_fut.save(f"{args.out_prefix}_futures.png")

    # Verification
    s = np.array(out_serial, dtype=np.int32)
    m = np.array(out_mp, dtype=np.int32)
    f = np.array(out_fut, dtype=np.int32)
    
    # Allow small tolerance for floating point differences in parallel execution
    match_mp = np.allclose(s, m, atol=1)
    match_fut = np.allclose(s, f, atol=1)
    
    print(f'MP matches Serial? {match_mp}')
    print(f'Futures matches Serial? {match_fut}')

if __name__ == '__main__':
    main()