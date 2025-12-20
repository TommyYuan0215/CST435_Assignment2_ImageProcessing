"""
Benchmark script to compare serial vs multiprocessing vs concurrent.futures pipelines.
Usage: python -m scripts.benchmark --input food-101-dataset/images --outdir out/bench
"""
import argparse
import csv
import os
import time
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from image_processing.filters import apply_pipeline
from image_processing.parallel_multiprocessing import apply_pipeline_multiprocessing
from image_processing.parallel_futures import apply_pipeline_futures

STEPS = [('grayscale', {}), ('gaussian', {}), ('sobel', {}), ('sharpen', {'alpha': 1.0})]

def collect_images(path: Path, sample: int):
    if path.is_file():
        return [path]
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    imgs = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(exts):
                imgs.append(Path(root) / f)
    imgs = sorted(imgs)
    if sample and sample > 0:
        imgs = imgs[:sample]
    return imgs

def run_trial_on_image(img_path: Path, resize: int, workers_list, trials, out_rows):
    img = Image.open(img_path).convert('RGB')
    if resize:
        w, h = img.size
        scale = resize / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

    # Serial
    for t in range(trials):
        t0 = time.perf_counter()
        _ = apply_pipeline(img, STEPS)
        elapsed = time.perf_counter() - t0
        out_rows.append({'image': str(img_path.name), 'pipeline': 'serial', 'workers': 1, 'trial': t, 'elapsed': elapsed})

    # Parallel
    for wcount in workers_list:
        # Multiprocessing
        for t in range(trials):
            t0 = time.perf_counter()
            _ = apply_pipeline_multiprocessing(img, STEPS, num_workers=wcount)
            elapsed = time.perf_counter() - t0
            out_rows.append({'image': str(img_path.name), 'pipeline': 'multiprocessing', 'workers': wcount, 'trial': t, 'elapsed': elapsed})

        # Futures
        for t in range(trials):
            t0 = time.perf_counter()
            _ = apply_pipeline_futures(img, STEPS, num_workers=wcount)
            elapsed = time.perf_counter() - t0
            out_rows.append({'image': str(img_path.name), 'pipeline': 'futures', 'workers': wcount, 'trial': t, 'elapsed': elapsed})

def save_csv(rows, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / 'benchmark_results.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'pipeline', 'workers', 'trial', 'elapsed'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return csv_path

def plot_results(rows, outdir: Path):
    agg = defaultdict(list)
    for r in rows:
        key = (r['pipeline'], int(r['workers']))
        agg[key].append(r['elapsed'])

    pipelines = ['serial', 'multiprocessing', 'futures']
    plt.figure(figsize=(8, 5))
    for p in pipelines:
        xs = []
        ys = []
        keys = sorted(k for k in agg.keys() if k[0] == p)
        for (_, w) in keys:
            xs.append(w)
            ys.append(np.mean(agg[(p, w)]))
        if xs:
            plt.plot(xs, ys, marker='o', label=p)
    plt.xlabel('Workers')
    plt.ylabel('Mean elapsed (s)')
    plt.title('Benchmark: pipeline runtime vs workers')
    plt.legend()
    plt.grid(True)
    outpng = outdir / 'benchmark_plot.png'
    plt.savefig(outpng, dpi=150)
    plt.close()
    return outpng

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input image file or directory')
    parser.add_argument('--outdir', '-o', required=True, help='Output directory for CSV and plots')
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4], help='Worker counts')
    parser.add_argument('--trials', type=int, default=3, help='Trials per configuration')
    parser.add_argument('--sample', type=int, default=5, help='Number of images to sample')
    parser.add_argument('--resize', type=int, default=256, help='Resize max dimension')
    args = parser.parse_args()

    imgs = collect_images(Path(args.input), args.sample)
    if not imgs:
        print('No images found for benchmarking')
        return

    workers_list = sorted(set([int(w) for w in args.workers if w > 0]))
    rows = []

    print(f'Benchmarking {len(imgs)} images, workers={workers_list}')
    for img_path in imgs:
        print('Running:', img_path.name)
        run_trial_on_image(img_path, args.resize, workers_list, args.trials, rows)

    outdir = Path(args.outdir)
    csv_path = save_csv(rows, outdir)
    png_path = plot_results(rows, outdir)
    print('Saved CSV:', csv_path)
    print('Saved plot:', png_path)

if __name__ == '__main__':
    main()