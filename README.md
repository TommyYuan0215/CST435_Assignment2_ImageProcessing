# Parallel Image Processing (Food-101 subset)

This project implements an image processing pipeline (Grayscale, Gaussian blur, Sobel edge detection, Sharpening, Brightness adjust) and compares parallelization approaches in Python: `multiprocessing` and `concurrent.futures` (process-based).

Quick start (local)

1. Create a subset of the Food-101 images (see `scripts/generate_subset.py`).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run serial pipeline (example):

```bash
python process_serial.py --input /path/to/subset --output out/serial
```

4. Run multiprocessing pipeline:

```bash
python process_mp.py --input /path --output out/mp --workers 4
```

5. Use `benchmark.py` to benchmark and generate CSV/plots. Example:

```bash
python benchmark.py --input ./subset --out ./out/bench --workers 1 2 4 --trials 3
```

This repository focuses on local Python implementations and benchmarking. For optional cloud deployment notes, see `docs/gcp.md`.
