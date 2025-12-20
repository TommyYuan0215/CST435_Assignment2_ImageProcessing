# Parallel Image Processing Pipeline (CST435 Assignment 2)

## Prepared By:
* **Tan Jun Lin** - 160989
* **Peh Jia Jin** - 161059
* **Ooi Tze Shen** - 165229
* **Wan Shan Jie** - 163836

This repository implements a high-performance image processing pipeline in Python for CST435. It compares a **Serial** pipeline against two **Parallel** paradigms: `multiprocessing` (Pool) and `concurrent.futures` (ProcessPoolExecutor). The pipeline applies a sequence of computationally intensive filters to images from the Food-101 dataset.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Assignment Details](#assignment-details)
- [Project Structure](#project-structure)
- [Notes & Tips](#notes--tips)

---

## 📋 Features

### Image Filters Implemented
The pipeline applies the following five operations, in order:

1. **Grayscale Conversion** — Uses the standard luminance formula: `Y = 0.299R + 0.587G + 0.114B`.
2. **Gaussian Blur (3x3)** — Smooths images using a 3x3 convolution kernel to reduce noise.
3. **Sobel Edge Detection** — Computes gradient magnitude using horizontal and vertical masks.
4. **Sharpening (Unsharp Masking)** — `result = original + alpha * (original - blurred)` to enhance edges.
5. **Brightness Adjustment** — Adjusts pixel values uniformly to increase/decrease brightness.

### Parallel Implementation

- **Split Strategy**: Images are split into horizontal chunks for parallel processing.
- **Overlap / Halo**: Chunks include overlap rows so convolution kernels (Gaussian, Sobel) do not produce boundary artifacts.
- **Implemented Paradigms**:
  - `multiprocessing` using `multiprocessing.Pool`.
  - `concurrent.futures` using `ProcessPoolExecutor`.

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd CST435_Assignment2_ImageProcessing
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

> All commands should be run from the project root directory.

### 1) Single Image Verification (`run_pipeline.py`)
Run the serial and both parallel pipelines on a single image to verify that outputs match:

```bash
python -m scripts.run_pipeline <input_image> <output_prefix> [--workers N]

# Example
python -m scripts.run_pipeline image.png out/test_result --workers 4
```

This produces files like `out/test_result_serial.png`, `out/test_result_mp.png`, `out/test_result_futures.png` for visual comparison.

### 2) Benchmarking (`benchmark.py`)
Run performance tests on a directory or single image.

```bash
python -m scripts.benchmark --input <path> --outdir <path> --workers 1 2 4 --sample 5
```

Key options:
- `--input`: Path to an image or directory of images.
- `--outdir`: Directory where `benchmark_results.csv` and `benchmark_plot.png` will be saved.
- `--workers`: A list of worker counts to test (e.g., `1 2 4 8`).
- `--trials`: Number of repetitions per config (default: 3).
- `--sample`: Number of images sampled from the input directory (default: 5).
- `--resize`: Resize images to a max dimension to speed benchmarking (default: 256; set `0` to disable).

Examples:

Quick test:
```bash
# Save results directly to the top-level `out/` directory
python -m scripts.benchmark --input food-101-dataset/images --outdir out --workers 1 2 4 --sample 5

# Or, use a sub-directory for this benchmark run (e.g. `out/bench`)
python -m scripts.benchmark --input food-101-dataset/images --outdir out/bench --workers 1 2 4 --sample 5
```

Full scaling test:
```bash
# Save results directly to the top-level `out/` directory
python -m scripts.benchmark --input food-101-dataset/images --outdir out --workers 1 4 8 12 16 20 --sample 20 --trials 5
```

**Note:** `--outdir` can be any directory; the script will create it if it does not already exist and will write `benchmark_results.csv` and `benchmark_plot.png` into that directory.

### 3) Run Tests
Ensure correctness (parallel outputs match serial):

```bash
pytest
```

---

## 📊 Performance Analysis
After running benchmarks, check `out/bench` for:

- `benchmark_results.csv` — Raw timing data for every trial.
- `benchmark_plot.png` — Visual speedup plot comparing parallel methods versus serial baseline.

---

## 📂 Project Structure

```text
CST435_Assignment2_ImageProcessing/
├── food-101-dataset/       # Dataset (images, metadata)
├── out/                    # Generated outputs (images, plots, CSVs)
├── scripts/                # Entry points
│   ├── benchmark.py        # Performance testing script
│   └── run_pipeline.py     # Single image verification script
├── image_processing/       # Core package
│   ├── filters.py          # Core filter implementations
│   ├── utils.py            # Array helpers, workers
│   ├── parallel_futures.py # concurrent.futures implementation
│   └── parallel_multiprocessing.py # multiprocessing implementation
├── test_filters.py         # Pytest unit tests
├── README.md               # This file
└── requirements.txt        # Dependencies
```

---

## 💡 Notes & Tips
- Use `--workers 1` as the serial baseline when benchmarking.
- The halo/overlap is critical for correct convolution across chunk boundaries — do not remove it.
- For reproducible benchmarking, fix random seeds where sampling is involved.

---
