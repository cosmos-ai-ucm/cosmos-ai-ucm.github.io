"""
Day 1 Lab: Serial vs Threads vs Processes vs NumPy

Run:

    python run_day1_lab.py

This lab compares:
1. Serial Python execution
2. ThreadPoolExecutor
3. ProcessPoolExecutor
4. NumPy vectorization
5. I/O-like tasks using sleep()

Important:
- ProcessPoolExecutor should be run inside if __name__ == "__main__"
- This is especially important on Windows/macOS and useful for portability.
"""

import os
import time
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

try:
    import numpy as np
except ImportError:
    np = None


# ------------------------------------------------------------
# Utility: Timing function
# ------------------------------------------------------------

def time_function(label, func, repeats=3):
    """
    Run func() several times and return timing statistics.

    Parameters
    ----------
    label : str
        Name of experiment.
    func : callable
        Function with no arguments.
    repeats : int
        Number of repeats.

    Returns
    -------
    dict
        Timing summary.
    """
    times = []

    for r in range(repeats):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

    return {
        "label": label,
        "min": min(times),
        "mean": statistics.mean(times),
        "max": max(times),
        "result": result,
    }


# ------------------------------------------------------------
# CPU-bound task
# ------------------------------------------------------------

def cpu_task(x):
    """
    A deliberately CPU-heavy pure Python function.

    This function does repeated modular arithmetic in Python.
    It is intentionally not vectorized, so it helps illustrate
    the difference between threads and processes.
    """
    total = 0
    for i in range(8_000):
        total += (x * i) % 97
    return total


def run_serial_cpu(data):
    return [cpu_task(x) for x in data]


def run_threaded_cpu(data, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(cpu_task, data))


def run_process_cpu(data, workers, chunksize=100):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(cpu_task, data, chunksize=chunksize))


# ------------------------------------------------------------
# I/O-like task
# ------------------------------------------------------------

def io_like_task(x):
    """
    Simulates an I/O-bound task.

    time.sleep() represents waiting for a file, network request,
    database response, etc.
    """
    time.sleep(0.01)
    return x * x


def run_serial_io(data):
    return [io_like_task(x) for x in data]


def run_threaded_io(data, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(io_like_task, data))


def run_process_io(data, workers):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(io_like_task, data, chunksize=10))


# ------------------------------------------------------------
# NumPy vectorized task
# ------------------------------------------------------------

def run_numpy_vectorized(n):
    """
    Vectorized numeric computation.

    This workload is not identical to cpu_task, but it demonstrates
    how array-based computation can outperform Python loops because
    NumPy runs optimized compiled operations.
    """
    if np is None:
        return None

    x = np.arange(n, dtype=np.int64)
    total = np.zeros_like(x)

    for i in range(8_000):
        total += (x * i) % 97

    return total


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------

def print_table(rows, baseline_time=None):
    """
    Pretty-print results.
    """
    print()
    print("=" * 88)
    print(f"{'Experiment':40s} {'Mean Time (s)':>15s} {'Speedup':>12s} {'Efficiency':>12s}")
    print("-" * 88)

    for row in rows:
        label = row["label"]
        mean_time = row["mean"]
        workers = row.get("workers", 1)

        if baseline_time is None:
            speedup = 1.0
        else:
            speedup = baseline_time / mean_time

        efficiency = speedup / workers

        print(f"{label:40s} {mean_time:15.4f} {speedup:12.2f} {efficiency:12.2f}")

    print("=" * 88)
    print()


def check_correctness(reference, candidate, label):
    """
    Check that two result lists are identical.
    """
    if reference == candidate:
        print(f"[OK] {label} matches serial result.")
    else:
        print(f"[ERROR] {label} does NOT match serial result!")


# ------------------------------------------------------------
# Main lab driver
# ------------------------------------------------------------

def main():
    print("Day 1 Lab: Serial vs Threads vs Processes vs NumPy")
    print(f"CPU count reported by os.cpu_count(): {os.cpu_count()}")

    repeats = 3

    # Keep data size moderate for a classroom setting.
    # You can increase this if machines are strong.
    cpu_data = list(range(20_000))
    io_data = list(range(200))

    worker_counts = [1, 2, 4, 8]

    if os.cpu_count() is not None and os.cpu_count() < 8:
        worker_counts = [w for w in worker_counts if w <= os.cpu_count()]

    # --------------------------------------------------------
    # Part A: CPU-bound benchmark
    # --------------------------------------------------------

    print("\nPART A: CPU-bound pure Python task")

    cpu_rows = []

    serial_cpu = time_function(
        "CPU serial",
        lambda: run_serial_cpu(cpu_data),
        repeats=repeats
    )
    serial_cpu["workers"] = 1
    cpu_rows.append(serial_cpu)

    reference_cpu_result = serial_cpu["result"]

    for workers in worker_counts:
        threaded = time_function(
            f"CPU threads, workers={workers}",
            lambda w=workers: run_threaded_cpu(cpu_data, w),
            repeats=repeats
        )
        threaded["workers"] = workers
        check_correctness(reference_cpu_result, threaded["result"], threaded["label"])
        cpu_rows.append(threaded)

    for workers in worker_counts:
        processed = time_function(
            f"CPU processes, workers={workers}",
            lambda w=workers: run_process_cpu(cpu_data, w, chunksize=100),
            repeats=repeats
        )
        processed["workers"] = workers
        check_correctness(reference_cpu_result, processed["result"], processed["label"])
        cpu_rows.append(processed)

    print_table(cpu_rows, baseline_time=serial_cpu["mean"])

    # --------------------------------------------------------
    # Part B: I/O-like benchmark
    # --------------------------------------------------------

    print("\nPART B: I/O-like task using sleep()")

    io_rows = []

    serial_io = time_function(
        "I/O serial",
        lambda: run_serial_io(io_data),
        repeats=repeats
    )
    serial_io["workers"] = 1
    io_rows.append(serial_io)

    reference_io_result = serial_io["result"]

    for workers in worker_counts + [16]:
        threaded_io = time_function(
            f"I/O threads, workers={workers}",
            lambda w=workers: run_threaded_io(io_data, w),
            repeats=repeats
        )
        threaded_io["workers"] = workers
        check_correctness(reference_io_result, threaded_io["result"], threaded_io["label"])
        io_rows.append(threaded_io)

    for workers in worker_counts:
        process_io = time_function(
            f"I/O processes, workers={workers}",
            lambda w=workers: run_process_io(io_data, w),
            repeats=repeats
        )
        process_io["workers"] = workers
        check_correctness(reference_io_result, process_io["result"], process_io["label"])
        io_rows.append(process_io)

    print_table(io_rows, baseline_time=serial_io["mean"])

    # --------------------------------------------------------
    # Part C: NumPy vectorization demonstration
    # --------------------------------------------------------

    print("\nPART C: NumPy vectorization demonstration")

    if np is not None:
        numpy_result = time_function(
            "NumPy vectorized",
            lambda: run_numpy_vectorized(20_000),
            repeats=repeats
        )
        numpy_result["workers"] = 1

        print_table([numpy_result], baseline_time=serial_cpu["mean"])
    else:
        print("NumPy is not installed. Skipping vectorized benchmark.")

    # --------------------------------------------------------
    # Student interpretation prompts
    # --------------------------------------------------------

    print("\nReflection Questions:")
    print("1. Did threads help for the CPU-bound task? Why or why not?")
    print("2. Did processes help for the CPU-bound task? Why or why not?")
    print("3. Did threads help for the I/O-like task? Why?")
    print("4. When did adding more workers stop helping?")
    print("5. Which result surprised you the most?")
    print("6. What does this imply for your final project?")


if __name__ == "__main__":
    main()