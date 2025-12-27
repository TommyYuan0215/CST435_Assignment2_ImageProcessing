"""
Combined Benchmark & Processing Script.
- Measures execution time for Serial, Multiprocessing, and Futures.
- SAVES ONLY ONE copy of the processed image (from Serial) for visual verification.
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

# Imports assume your library files use the "Global Pool" logic we discussed
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

def load_and_resize(img_path: Path, resize: int):
    img = Image.open(img_path).convert('RGB')
    if resize and resize > 0:
        w, h = img.size
        scale = resize / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
    return img

def save_single_result(img, folder: Path, original_name):
    """
    Saves a single processed image for verification.
    Location: out/processed_image/image_name.png
    """
    folder.mkdir(parents=True, exist_ok=True)
    clean_name = Path(original_name).stem
    # Simple filename, no pipeline info needed since we only save one
    filename = f"{clean_name}_processed.png"
    save_path = folder / filename
    img.save(save_path)

def run_serial_benchmark(imgs, resize, trials, out_rows, save_dir):
    print("Running Serial Baseline...")
    for i, img_path in enumerate(imgs):
        img = load_and_resize(img_path, resize)
        for t in range(trials):
            t0 = time.perf_counter()
            # 1. Process
            out_img = apply_pipeline(img, STEPS)
            # 2. Stop Timer
            elapsed = time.perf_counter() - t0
            
            # 3. Save only the first processed image for verification
            if t == 0 and save_dir:
                save_single_result(out_img, save_dir, img_path.name)

            out_rows.append({
                'image': str(img_path.name), 'pipeline': 'serial', 
                'workers': 1, 'trial': t, 'elapsed': elapsed
            })

def run_multiprocessing_benchmark(imgs, resize, workers_list, trials, out_rows):
    for wcount in workers_list:
        print(f"Running Multiprocessing (Workers={wcount})...")
        for img_path in imgs:
            img = load_and_resize(img_path, resize)
            for t in range(trials):
                t0 = time.perf_counter()
                # 1. Process (No saving, pure benchmark)
                _ = apply_pipeline_multiprocessing(img, STEPS, num_workers=wcount)
                # 2. Stop Timer
                elapsed = time.perf_counter() - t0

                out_rows.append({
                    'image': str(img_path.name), 'pipeline': 'multiprocessing', 
                    'workers': wcount, 'trial': t, 'elapsed': elapsed
                })

def run_futures_benchmark(imgs, resize, workers_list, trials, out_rows):
    for wcount in workers_list:
        print(f"Running Futures (Workers={wcount})...")
        for img_path in imgs:
            img = load_and_resize(img_path, resize)
            for t in range(trials):
                t0 = time.perf_counter()
                # 1. Process (No saving, pure benchmark)
                _ = apply_pipeline_futures(img, STEPS, num_workers=wcount)
                # 2. Stop Timer
                elapsed = time.perf_counter() - t0

                out_rows.append({
                    'image': str(img_path.name), 'pipeline': 'futures', 
                    'workers': wcount, 'trial': t, 'elapsed': elapsed
                })

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
    plt.figure(figsize=(10, 6))
    for p in pipelines:
        xs = []
        ys = []
        keys = sorted(k for k in agg.keys() if k[0] == p)
        for (_, w) in keys:
            xs.append(w)
            ys.append(np.mean(agg[(p, w)]))
        if xs:
            plt.plot(xs, ys, marker='o', label=p, linewidth=2)
    plt.xlabel('Workers', fontsize=12)
    plt.ylabel('Mean elapsed (s)', fontsize=12)
    plt.title('Benchmark: Pipeline Runtime vs Workers', fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    outpng = outdir / 'benchmark_plot.png'
    plt.savefig(outpng, dpi=150)
    plt.close()
    return outpng

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='food-101-dataset/images', help='Input image file or directory')
    parser.add_argument('--outdir', '-o', default='out', help='Output directory for CSV and plots')
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4], help='Worker counts')
    parser.add_argument('--trials', type=int, default=1, help='Trials per configuration')
    parser.add_argument('--sample', type=int, default=5, help='Number of images to sample')
    parser.add_argument('--resize', type=int, default=0, help='Resize max dimension (0=Original Size)')
    args = parser.parse_args()

    imgs = collect_images(Path(args.input), args.sample)
    if not imgs:
        print('No images found for benchmarking')
        return

    # Define output directory for images
    outdir_base = Path(args.outdir)
    save_dir = outdir_base / 'processed_image'
    
    workers_list = sorted(set([int(w) for w in args.workers if w > 0]))
    rows = []

    print(f'Benchmarking {len(imgs)} images. Workers={workers_list}')
    print(f'One sample processed image will be saved to: {save_dir}')

    # 1. Run Serial (Saves the sample image)
    run_serial_benchmark(imgs, args.resize, args.trials, rows, save_dir)
    
    # 2. Run Futures (Pure benchmark)
    run_futures_benchmark(imgs, args.resize, workers_list, args.trials, rows)

    # 3. Run Multiprocessing (Pure benchmark)
    run_multiprocessing_benchmark(imgs, args.resize, workers_list, args.trials, rows)

    csv_path = save_csv(rows, outdir_base)
    png_path = plot_results(rows, outdir_base)
    print(f'\nDone! Results saved to {args.outdir}')
    print(f'CSV: {csv_path}')
    print(f'Plot: {png_path}')
    print(f'Sample Image: {save_dir}')

if __name__ == '__main__':
    main()