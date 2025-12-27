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

---

## Features

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
  - `multiprocessing`: Uses `multiprocessing.Pool`.
  - `concurrent.futures`: Uses `ThreadPoolExecutor`.

---

## Installation

1. Install Python Environment
Before running the project, ensure you have Python installed on your system. This project requires Python 3.10 or higher.

- For Windows Users:
  1. Download: Go to the official [Python Downloads page](https://www.python.org/downloads/windows/)
  2. Install: Run the installer and crucially check the box that says "Add python.exe to PATH" before clicking "Install Now."
  3. Verify: Open PowerShell or Command Prompt and type:

  ```bash
  python --version
  ```

- For macOS Users:
  - Install Python using Homebrew:

  ```bash
  brew install python
  ```
  - Or download the installer from python.org.

- For Linux Users:
  - Use your distribution's package manager to install Python and the virtual environment module:

  ```bash
  # Ubuntu/Debian/Mint
  sudo apt update
  sudo apt install python3 python3-venv python3-pip

  # Fedora
  sudo dnf install python3
  ```

2. Clone the repository:

```bash
git clone <repository_url>
cd CST435_Assignment2_ImageProcessing
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
# For macOS / Linux user
source .venv/bin/activate
```

```bash
# For Windows user
.venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```
---

## Dataset Download
This project uses the Food-101 dataset. You must download and extract it manually before running the benchmark.

1. Download: Go to the dataset page on Kaggle: https://www.kaggle.com/datasets/dansbecker/food-101
2. Extract: Unzip the downloaded file into the root directory of this project.
3. Rename: Ensure the extracted folder is named food-101-dataset.

Your directory structure should look like this:

```text
CST435_Assignment2_ImageProcessing/
├── food-101-dataset/       <-- Extracted dataset folder
│   ├── images/             <-- Contains the image subfolders
│   └── meta/
├── image_processing/
└── ...
```

---

## Usage

> All commands should be run from the project root directory..

### 1) Performance Benchmarking (Speedup Analysis)

**Script:** `main.py`

**Purpose:** Runs performance tests on a directory of images or a single image using multiple worker counts (e.g., 1, 2, 4, 8) to compare execution times. This is essential for the Performance Analysis section of your technical report.

**Command:**

```bash
python main.py --workers <list> --sample <N>
```

**Example:**

```bash
python main.py --workers 1 2 4 --sample 5
```

**Output:**

- `benchmark_results.csv`: Contains raw timing data for every trial.
- `benchmark_plot.png`: A visual plot showing the speedup of parallel methods versus the serial baseline.
- `processed_image/`: Folder containing the processed output images from the benchmark run.

Key options:
- `--input`: Path to an image or directory of images.
- `--outdir`: Directory where `benchmark_results.csv` and `benchmark_plot.png` will be saved.
- `--workers`: A list of worker counts to test (default: [1, 2, 4]).
- `--trials`: Number of repetitions per config (default: 1).
- `--sample`: Number of images sampled from the input directory (default: 5).
- `--resize`: Resize images to a max dimension to speed benchmarking (default: 0， which will be use original size).
- `--sample`: When set to 0, process *all* images found under the input directory (no sampling). Use with care for full-dataset runs.

**Note:** `--outdir` can be any directory; the script will create it if it does not already exist and will write `benchmark_results.csv` and `benchmark_plot.png` into that directory.

---

## Performance Analysis
After running benchmarks, check the output directory you passed to `--outdir` (for example `out/` or `out/bench`) for:

- `benchmark_results.csv` — Raw timing data for every trial.
- `benchmark_plot.png` — Visual speedup plot comparing parallel methods versus serial baseline.

If you ran a full dataset benchmark (e.g., `--sample 0`) the CSV may be large; consider compressing or sampling the CSV when analyzing results.

---

## Project Structure

```text
CST435_Assignment2_ImageProcessing/
├── food-101-dataset/       # Dataset (images, metadata) — use `scripts/download_dataset.py` to fetch
├── out/                    # Generated outputs (images, plots, CSVs)
├── main.py                 # Main entry point (Benchmark & Performance testing)
├── image_processing/       # Core package
│   ├── filters.py          # Core filter implementations
│   ├── utils.py            # Array helpers, workers
│   ├── parallel_futures.py # concurrent.futures implementation
│   └── parallel_multiprocessing.py # multiprocessing implementation
├── README.md               # This file
└── requirements.txt        # Dependencies
```

> Note: If you already downloaded the dataset by other means, ensure the top-level `food-101-dataset/images/` path exists before running benchmarks.

---

## Notes & Tips
- **Serial Baseline:** Use `--workers 1` as the serial baseline when benchmarking to see the pure processing speed without overhead.
- **Worker Counts:** For parallel paradigms, we recommend testing with **2 or more workers** to observe actual speedup. A single parallel worker is often slower than the serial implementation due to initialization overhead.
- **Overlap:** The halo/overlap logic is critical for correct convolution across chunk boundaries — do not remove it.
---
