"""
Day 2 Lab: Threaded Dataset Loader

Run:

    python day2_threaded_loader.py

This lab compares:
1. Serial simulated file loading
2. ThreadPoolExecutor-based loading
3. Different thread counts
4. Throughput and speedup
5. Error handling for corrupted files

The simulated loader uses time.sleep() to mimic I/O waiting.
"""

import time
import random
import statistics
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

NUM_FILES = 500
RANDOM_SEED = 42
WORKER_COUNTS = [1, 2, 4, 8, 16, 32]
ERROR_RATE = 0.03

random.seed(RANDOM_SEED)


# ------------------------------------------------------------
# Simulated I/O-bound loader
# ------------------------------------------------------------

def simulated_load(path):
    """
    Simulate loading a file from disk.

    This function imitates an I/O-bound AI data loading task:
    - waiting for disk/network
    - occasional corrupted input
    - returning image-like metadata

    Parameters
    ----------
    path : str
        Fake file path.

    Returns
    -------
    dict
        Simulated metadata.
    """

    # Simulate variable file-read latency
    time.sleep(random.uniform(0.01, 0.04))

    # Simulate occasional corrupted files
    if random.random() < ERROR_RATE:
        raise ValueError(f"corrupted file: {path}")

    # Return fake metadata similar to an image dataset
    return {
        "path": path,
        "width": random.choice([128, 224, 256, 384, 512]),
        "height": random.choice([128, 224, 256, 384, 512]),
        "brightness": random.random(),
        "label": random.choice(["cat", "dog", "car", "bird"]),
    }


# ------------------------------------------------------------
# Serial loader
# ------------------------------------------------------------

def load_serial(paths):
    """
    Load paths one at a time.

    This is the baseline version.
    """

    results = []
    errors = []

    for path in paths:
        try:
            record = simulated_load(path)
            results.append(record)
        except Exception as e:
            errors.append({
                "path": path,
                "error": str(e),
            })

    return results, errors


# ------------------------------------------------------------
# ThreadPoolExecutor loader
# ------------------------------------------------------------

def load_threaded(paths, workers):
    """
    Load paths using a thread pool.

    This version uses submit + as_completed so that:
    - results are processed as soon as they finish
    - exceptions can be handled per file
    - slow files do not block fast files
    """

    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(simulated_load, path): path
            for path in paths
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]

            try:
                record = future.result()
                results.append(record)
            except Exception as e:
                errors.append({
                    "path": path,
                    "error": str(e),
                })

    return results, errors


# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------

def summarize_metadata(records):
    """
    Compute simple dataset statistics.
    """

    if not records:
        return {
            "count": 0,
            "avg_width": None,
            "avg_height": None,
            "avg_brightness": None,
        }

    avg_width = sum(r["width"] for r in records) / len(records)
    avg_height = sum(r["height"] for r in records) / len(records)
    avg_brightness = sum(r["brightness"] for r in records) / len(records)

    label_counts = {}
    for r in records:
        label = r["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "count": len(records),
        "avg_width": avg_width,
        "avg_height": avg_height,
        "avg_brightness": avg_brightness,
        "label_counts": label_counts,
    }


# ------------------------------------------------------------
# Benchmarking helpers
# ------------------------------------------------------------

def time_run(func, repeats=3):
    """
    Run a function several times and return timing statistics.
    """

    times = []
    last_result = None

    for _ in range(repeats):
        start = time.perf_counter()
        last_result = func()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "max": max(times),
        "result": last_result,
    }


def benchmark(paths, repeats=3):
    """
    Run serial and threaded benchmarks.
    """

    rows = []

    print("\nRunning serial baseline...")

    serial_timing = time_run(
        lambda: load_serial(paths),
        repeats=repeats
    )

    serial_results, serial_errors = serial_timing["result"]
    serial_time = serial_timing["mean"]

    rows.append({
        "method": "serial",
        "workers": 1,
        "mean_time": serial_time,
        "speedup": 1.0,
        "throughput": len(paths) / serial_time,
        "loaded": len(serial_results),
        "errors": len(serial_errors),
    })

    print(f"Serial mean time: {serial_time:.3f} s")

    for workers in WORKER_COUNTS:
        print(f"\nRunning threaded loader with workers={workers}...")

        timing = time_run(
            lambda w=workers: load_threaded(paths, workers=w),
            repeats=repeats
        )

        results, errors = timing["result"]
        mean_time = timing["mean"]
        speedup = serial_time / mean_time
        throughput = len(paths) / mean_time

        rows.append({
            "method": "threaded",
            "workers": workers,
            "mean_time": mean_time,
            "speedup": speedup,
            "throughput": throughput,
            "loaded": len(results),
            "errors": len(errors),
        })

        print(
            f"workers={workers:2d} | "
            f"time={mean_time:.3f} s | "
            f"speedup={speedup:.2f}x | "
            f"throughput={throughput:.1f} files/s | "
            f"errors={len(errors)}"
        )

    return rows


def save_results_csv(rows, filename="results.csv"):
    """
    Save benchmark rows to a CSV file.
    """

    fieldnames = [
        "method",
        "workers",
        "mean_time",
        "speedup",
        "throughput",
        "loaded",
        "errors",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {filename}")


def print_results_table(rows):
    """
    Print a clean results table.
    """

    print()
    print("=" * 92)
    print(
        f"{'Method':12s} "
        f"{'Workers':>8s} "
        f"{'Mean Time (s)':>15s} "
        f"{'Speedup':>10s} "
        f"{'Throughput':>15s} "
        f"{'Loaded':>8s} "
        f"{'Errors':>8s}"
    )
    print("-" * 92)

    for row in rows:
        print(
            f"{row['method']:12s} "
            f"{row['workers']:8d} "
            f"{row['mean_time']:15.3f} "
            f"{row['speedup']:10.2f} "
            f"{row['throughput']:15.1f} "
            f"{row['loaded']:8d} "
            f"{row['errors']:8d}"
        )

    print("=" * 92)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("Day 2 Lab: Threaded Dataset Loader")
    print(f"Number of simulated files: {NUM_FILES}")
    print(f"Worker counts: {WORKER_COUNTS}")

    paths = [f"image_{i:05d}.jpg" for i in range(NUM_FILES)]

    rows = benchmark(paths, repeats=3)

    print_results_table(rows)
    save_results_csv(rows)

    # Run one final threaded load and summarize metadata
    results, errors = load_threaded(paths, workers=8)
    summary = summarize_metadata(results)

    print("\nExample dataset summary using workers=8:")
    print(summary)

    print("\nReflection questions:")
    print("1. Which worker count gave the best throughput?")
    print("2. When did speedup start to flatten?")
    print("3. Why might too many threads hurt performance?")
    print("4. How were corrupted files handled?")
    print("5. Where could this pattern help your final project?")


if __name__ == "__main__":
    main()