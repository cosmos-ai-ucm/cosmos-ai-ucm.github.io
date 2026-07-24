"""
Day 3 Lab: Multiprocessing for CPU-Bound AI Preprocessing

Run:

    python day3_multiprocessing_lab.py

This lab compares:
1. Serial CPU-heavy feature extraction
2. Item-level multiprocessing with ProcessPoolExecutor
3. Chunk-level map-reduce multiprocessing
4. Different worker counts
5. Different chunk sizes

Important:
- Worker functions must be defined at the top level.
- ProcessPoolExecutor code should be protected by:

      if __name__ == "__main__":
          main()
"""

import os
import time
import csv
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_SIZE = 40_000
CPU_WORK = 30_000
REPEATS = 3

WORKER_COUNTS = [1, 2, 4, 8]
CHUNK_SIZES = [100, 500, 1000, 5000]


# ------------------------------------------------------------
# CPU-heavy feature extraction
# ------------------------------------------------------------

def cpu_feature(x):
    """
    Simulate CPU-heavy feature extraction.

    This represents Python-level preprocessing such as:
    - tokenization
    - image-statistics extraction
    - handcrafted feature computation
    - expensive row-wise tabular transformation
    """

    total = 0

    for i in range(CPU_WORK):
        total += (x * i) % 97

    return total


def make_data(n=DATA_SIZE):
    """
    Create synthetic input data.

    In a real project, this could represent:
    - image file IDs
    - text document IDs
    - tabular row IDs
    - user IDs
    """

    return list(range(n))


# ------------------------------------------------------------
# Serial baseline
# ------------------------------------------------------------

def run_serial(data):
    """
    Serial baseline.

    Always implement this first so we can compare correctness
    and speedup.
    """

    return [cpu_feature(x) for x in data]


# ------------------------------------------------------------
# Item-level multiprocessing
# ------------------------------------------------------------

def run_parallel_item_level(data, workers, chunksize=100):
    """
    Process each item independently using ProcessPoolExecutor.map.

    This is simple, but if individual tasks are too small,
    scheduling and serialization overhead can dominate.
    """

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                cpu_feature,
                data,
                chunksize=chunksize,
            )
        )

    return results


# ------------------------------------------------------------
# Chunk-level map-reduce
# ------------------------------------------------------------

def chunk_data(data, chunk_size):
    """
    Split data into chunks.
    """

    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def process_chunk(chunk):
    """
    Map step.

    Each process receives a chunk, computes local features,
    and returns a compact partial result.

    Returning a small partial result can be more efficient than
    returning a huge list of per-item outputs.
    """

    partial_sum = 0
    partial_count = 0
    partial_min = None
    partial_max = None

    for x in chunk:
        feature = cpu_feature(x)

        partial_sum += feature
        partial_count += 1

        if partial_min is None or feature < partial_min:
            partial_min = feature

        if partial_max is None or feature > partial_max:
            partial_max = feature

    return {
        "sum": partial_sum,
        "count": partial_count,
        "min": partial_min,
        "max": partial_max,
    }


def reduce_partials(partials):
    """
    Reduce step.

    Combine partial results from all workers.
    """

    total_sum = sum(p["sum"] for p in partials)
    total_count = sum(p["count"] for p in partials)

    global_min = min(p["min"] for p in partials if p["min"] is not None)
    global_max = max(p["max"] for p in partials if p["max"] is not None)

    mean = total_sum / total_count if total_count else None

    return {
        "sum": total_sum,
        "count": total_count,
        "mean": mean,
        "min": global_min,
        "max": global_max,
    }


def run_parallel_map_reduce(data, workers, chunk_size):
    """
    Chunk-level map-reduce multiprocessing.
    """

    chunks = list(chunk_data(data, chunk_size))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        partials = list(executor.map(process_chunk, chunks))

    return reduce_partials(partials)


# ------------------------------------------------------------
# Timing utilities
# ------------------------------------------------------------

def time_function(func, repeats=REPEATS):
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


def print_table(rows):
    """
    Print benchmark table.
    """

    print()
    print("=" * 110)
    print(
        f"{'Method':24s} "
        f"{'Workers':>8s} "
        f"{'Chunk':>8s} "
        f"{'Mean Time':>12s} "
        f"{'Speedup':>10s} "
        f"{'Efficiency':>12s} "
        f"{'Items/sec':>12s} "
        f"{'Correct':>10s}"
    )
    print("-" * 110)

    for row in rows:
        print(
            f"{row['method']:24s} "
            f"{row['workers']:8d} "
            f"{row['chunk_size']:8d} "
            f"{row['mean_time']:12.3f} "
            f"{row['speedup']:10.2f} "
            f"{row['efficiency']:12.2f} "
            f"{row['items_per_sec']:12.1f} "
            f"{str(row['correct']):>10s}"
        )

    print("=" * 110)


def save_results_csv(rows, filename="results.csv"):
    """
    Save benchmark results.
    """

    fieldnames = [
        "method",
        "workers",
        "chunk_size",
        "mean_time",
        "speedup",
        "efficiency",
        "items_per_sec",
        "correct",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {filename}")


# ------------------------------------------------------------
# Correctness helpers
# ------------------------------------------------------------

def summarize_serial_result(serial_result):
    """
    Convert full serial output into the same summary used by map-reduce.
    """

    total_sum = sum(serial_result)
    total_count = len(serial_result)

    return {
        "sum": total_sum,
        "count": total_count,
        "mean": total_sum / total_count,
        "min": min(serial_result),
        "max": max(serial_result),
    }


def summaries_match(a, b):
    """
    Compare summary dictionaries.
    """

    return (
        a["sum"] == b["sum"]
        and a["count"] == b["count"]
        and abs(a["mean"] - b["mean"]) < 1e-9
        and a["min"] == b["min"]
        and a["max"] == b["max"]
    )


# ------------------------------------------------------------
# Benchmark experiments
# ------------------------------------------------------------

def benchmark_item_level(data, serial_result, serial_time):
    """
    Benchmark item-level parallel processing.
    """

    rows = []

    for workers in WORKER_COUNTS:
        for chunksize in [1, 10, 100, 1000]:
            label = f"item-level workers={workers}, chunksize={chunksize}"

            print(f"\nRunning {label}")

            timing = time_function(
                lambda w=workers, c=chunksize: run_parallel_item_level(
                    data,
                    workers=w,
                    chunksize=c,
                )
            )

            result = timing["result"]
            mean_time = timing["mean"]

            correct = result == serial_result
            speedup = serial_time / mean_time
            efficiency = speedup / workers
            items_per_sec = len(data) / mean_time

            rows.append({
                "method": "item_level",
                "workers": workers,
                "chunk_size": chunksize,
                "mean_time": mean_time,
                "speedup": speedup,
                "efficiency": efficiency,
                "items_per_sec": items_per_sec,
                "correct": correct,
            })

    return rows


def benchmark_map_reduce(data, serial_summary, serial_time):
    """
    Benchmark chunk-level map-reduce processing.
    """

    rows = []

    for workers in WORKER_COUNTS:
        for chunk_size in CHUNK_SIZES:
            label = f"map-reduce workers={workers}, chunk={chunk_size}"

            print(f"\nRunning {label}")

            timing = time_function(
                lambda w=workers, c=chunk_size: run_parallel_map_reduce(
                    data,
                    workers=w,
                    chunk_size=c,
                )
            )

            summary = timing["result"]
            mean_time = timing["mean"]

            correct = summaries_match(serial_summary, summary)
            speedup = serial_time / mean_time
            efficiency = speedup / workers
            items_per_sec = len(data) / mean_time

            rows.append({
                "method": "map_reduce",
                "workers": workers,
                "chunk_size": chunk_size,
                "mean_time": mean_time,
                "speedup": speedup,
                "efficiency": efficiency,
                "items_per_sec": items_per_sec,
                "correct": correct,
            })

    return rows


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("Day 3 Lab: Multiprocessing for CPU-Bound AI Preprocessing")
    print(f"CPU count reported by os.cpu_count(): {os.cpu_count()}")
    print(f"Data size: {DATA_SIZE}")
    print(f"CPU_WORK per item: {CPU_WORK}")
    print(f"Worker counts: {WORKER_COUNTS}")
    print(f"Chunk sizes: {CHUNK_SIZES}")

    data = make_data(DATA_SIZE)

    print("\nRunning serial baseline...")

    serial_timing = time_function(lambda: run_serial(data))
    serial_result = serial_timing["result"]
    serial_time = serial_timing["mean"]
    serial_summary = summarize_serial_result(serial_result)

    print(f"Serial mean time: {serial_time:.3f} seconds")
    print("Serial summary:", serial_summary)

    rows = []

    rows.append({
        "method": "serial",
        "workers": 1,
        "chunk_size": 0,
        "mean_time": serial_time,
        "speedup": 1.0,
        "efficiency": 1.0,
        "items_per_sec": len(data) / serial_time,
        "correct": True,
    })

    # Experiment 1: item-level multiprocessing
    item_rows = benchmark_item_level(
        data=data,
        serial_result=serial_result,
        serial_time=serial_time,
    )

    # Experiment 2: chunk-level map-reduce
    map_reduce_rows = benchmark_map_reduce(
        data=data,
        serial_summary=serial_summary,
        serial_time=serial_time,
    )

    rows.extend(item_rows)
    rows.extend(map_reduce_rows)

    print_table(rows)
    save_results_csv(rows)

    print("\nReflection Questions:")
    print("1. Which approach was fastest: item-level or map-reduce?")
    print("2. Which worker count gave the best speedup?")
    print("3. Which chunk size worked best?")
    print("4. When did speedup flatten?")
    print("5. What overheads might explain your results?")
    print("6. How could this apply to your final project?")


if __name__ == "__main__":
    main()