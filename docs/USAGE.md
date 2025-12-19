## Quick Usage (new scripts)

- Run the combined pipeline (serial + two parallel versions) and compare outputs:

```bash
python scripts/run_pipeline.py input.jpg out_prefix --workers 4
```

- Run tests:

```bash
pytest -q
```

Notes:
- Filters: **Grayscale**, **3x3 Gaussian Blur**, **Sobel edges**, **Sharpen**, **Brightness Adjustment**
- Parallel paradigms: `multiprocessing` and `concurrent.futures.ProcessPoolExecutor`

Benchmarking:

Use `scripts/benchmark.py` to measure runtimes across worker counts. Example:

```bash
python scripts/benchmark.py --input ./subset --outdir out/bench --workers 1 2 4 8 --trials 3 --sample 10 --resize 256
```

This will save `benchmark_results.csv` and `benchmark_plot.png` into `out/bench`.
