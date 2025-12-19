# Parallel Image Processing Pipeline (CST435 Assignment 2)

This project implements a high-performance image processing pipeline in Python, designed to compare **Serial** execution against two **Parallel** paradigms: `multiprocessing` (shared memory/spawning) and `concurrent.futures` (process pool). 

The pipeline performs a sequence of computationally intensive filters on images from the Food-101 dataset.

## 📋 Features

### Image Filters Implemented
The pipeline applies the following 5 operations in order:
1.  **Grayscale Conversion**: Uses standard luminance formula ($Y = 0.299R + 0.587G + 0.114B$).
2.  **Gaussian Blur (3x3)**: Smooths image using a convolution kernel to reduce noise.
3.  **Sobel Edge Detection**: Computes gradient magnitude using horizontal and vertical masks.
4.  **Sharpening**: Enhances edges using an unsharp masking technique (`original + alpha * (original - blurred)`).
5.  **Brightness Adjustment**: Direct pixel value modification.

### Parallel Implementation
* **Split Strategy**: Images are split into horizontal chunks.
* **Overlap/Halo**: Implemented smart overlap (padding) between chunks to ensure convolution filters (Gaussian, Sobel) have no artifacts at chunk boundaries.
* **Paradigms**:
    * `multiprocessing`: Uses `multiprocessing.Pool`.
    * `concurrent.futures`: Uses `ProcessPoolExecutor`.

## 🛠️ Installation

1.  **Clone the repository** (if needed):
    ```bash
    git clone <repository_url>
    cd CST435_Assignment2_ImageProcessing
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Run Pipeline on a Single Image
Run the full pipeline (Serial, Multiprocessing, and Futures) on one image to visually verify results.

```bash
python -m scripts.run_pipeline image_677963.png out/test_result --workers 4
```

Input: Path to your image (e.g., image_677963.png).

Output: Saves test_result_serial.png, test_result_mp.png, etc. to out/ folder.

### 2. Standard Benchmarking
Compare performance across typical worker counts (e.g., 1, 2, 4 CPUs).

```bash
python -m scripts.benchmark --input food-101-dataset/images --outdir out/bench --workers 1 2 4 --sample 10
```

### 3. Advanced High-Performance Benchmarking
To test the full scalability of the system on high-performance CPUs (e.g., 12 cores / 20 threads), use the following command:

```bash
python -m scripts.benchmark --input food-101-dataset/images --outdir out/bench --workers 1 4 8 12 16 20 --sample 20
```

Command Breakdown:

`python -m scripts.benchmark`: Runs the benchmark script as a module to correctly handle imports.

`--input`: Recursively searches the dataset directory.

`--workers 1 4 8 12 16 20`: Tests specific milestones:

- 1: Serial baseline.
- 4, 8: Linear scaling check.
- 12: Physical core limit.
- 16, 20: Logical thread limit (Hyper-threading).

`--sample 20`: Uses 20 random images for a statistically significant test.

### 4. Run Tests
Verify that parallel outputs mathematically match serial outputs.

```bash
pytest
```

## � Performance Analysis
After running the benchmark script, check the out/bench folder for:

- `benchmark_results.csv`: Raw timing data for every trial.
- `benchmark_plot.png`: A graph visualizing the speedup of Parallel methods vs. Serial as worker count increases.

## 📝 Assignment Details
Course: CST435

Option Selected: Python Implementation

Requirements Met:

- Implemented all 5 specific filters.
- Implemented multiprocessing module.
- Implemented concurrent.futures module.

## �📂 Project Structure

```text
CST435_Assignment2_ImageProcessing/
├── food-101-dataset/       # Dataset directory
├── out/                    # Generated outputs (images, plots, CSVs)
├── scripts/                # Entry points
│   ├── benchmark.py        # Performance testing script
│   └── run_pipeline.py     # Single image verification script
├── src/
│   └── image_processing/
│       ├── filters.py      # Core filter logic (Grayscale, Sobel, etc.)
│       ├── parallel_futures.py        # Implementation using concurrent.futures
│       └── parallel_multiprocessing.py # Implementation using multiprocessing
├── test_filters.py         # Pytest unit tests
├── README.md               # Documentation
└── requirements.txt        # Project dependencies
```