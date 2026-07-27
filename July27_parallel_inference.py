"""
Day 4 Lab: Parallel ML Inference and Evaluation

Run:

    python day4_parallel_inference.py

This lab compares:
1. Serial inference
2. Threaded sharded inference
3. Process-based sharded inference
4. Different worker counts
5. Different chunk sizes
6. Accuracy and correctness against serial baseline

Important:
- ProcessPoolExecutor code should be protected by:

      if __name__ == "__main__":
          main()

- For this lab, the model is trained with n_jobs=1 to avoid nested parallelism.
"""

import time
import csv
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed

import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RANDOM_SEED = 42
N_ESTIMATORS = 100

WORKER_COUNTS = [1, 2, 4, 8]
CHUNK_SIZES = [16, 64, 256]
REPEATS = 3


# ------------------------------------------------------------
# Data and model
# ------------------------------------------------------------

def make_dataset():
    """
    Load the scikit-learn digits dataset and split train/test data.
    """

    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a small classifier.

    We set n_jobs=1 to avoid hidden parallelism inside the model.
    This makes our external threading/process experiments easier
    to interpret.
    """

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    model.fit(X_train, y_train)
    return model


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
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "max": max(times),
        "result": last_result,
    }


# ------------------------------------------------------------
# Serial prediction baseline
# ------------------------------------------------------------

def serial_predict(model, X):
    """
    Baseline inference: predict all samples at once.
    """

    return model.predict(X)


# ------------------------------------------------------------
# Sharding helpers
# ------------------------------------------------------------

def make_chunks(X, chunk_size):
    """
    Split an array into chunks.

    Each chunk stores:
    - chunk_id
    - start index
    - end index
    - X slice

    Keeping start/end lets us reconstruct predictions in the
    original sample order.
    """

    chunks = []

    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))

        chunks.append({
            "chunk_id": len(chunks),
            "start": start,
            "end": end,
            "X": X[start:end],
        })

    return chunks


def predict_chunk(args):
    """
    Worker function for predicting one chunk.

    Must be top-level so it can be used by ProcessPoolExecutor.
    """

    model, chunk = args

    preds = model.predict(chunk["X"])

    return {
        "chunk_id": chunk["chunk_id"],
        "start": chunk["start"],
        "end": chunk["end"],
        "preds": preds,
    }


def combine_chunk_predictions(outputs, n_samples):
    """
    Combine chunk predictions into one prediction array.

    This preserves the original ordering using each chunk's
    start and end indices.
    """

    y_pred = np.empty(n_samples, dtype=int)

    for out in outputs:
        y_pred[out["start"]:out["end"]] = out["preds"]

    return y_pred


# ------------------------------------------------------------
# Threaded inference
# ------------------------------------------------------------

def threaded_predict(model, X, workers=4, chunk_size=64):
    """
    Parallel inference with ThreadPoolExecutor.

    Threads can be useful if the underlying model code releases
    the GIL or if inference includes I/O-like work.
    """

    chunks = make_chunks(X, chunk_size)
    tasks = [(model, chunk) for chunk in chunks]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outputs = list(executor.map(predict_chunk, tasks))

    return combine_chunk_predictions(outputs, len(X))


# ------------------------------------------------------------
# Process-based inference
# ------------------------------------------------------------

def process_predict(model, X, workers=4, chunk_size=64):
    """
    Parallel inference with ProcessPoolExecutor.

    Processes can help for CPU-heavy independent work, but may
    introduce overhead because the model and data must be sent
    to worker processes.
    """

    chunks = make_chunks(X, chunk_size)
    tasks = [(model, chunk) for chunk in chunks]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        outputs = list(
            executor.map(
                predict_chunk,
                tasks,
                chunksize=1,
            )
        )

    return combine_chunk_predictions(outputs, len(X))


# ------------------------------------------------------------
# Alternative: submit + as_completed
# ------------------------------------------------------------

def threaded_predict_as_completed(model, X, workers=4, chunk_size=64):
    """
    Alternative threaded implementation using submit + as_completed.

    This is useful when chunks have variable runtimes or when we
    want explicit exception handling.
    """

    chunks = make_chunks(X, chunk_size)
    outputs = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_chunk = {
            executor.submit(predict_chunk, (model, chunk)): chunk
            for chunk in chunks
        }

        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]

            try:
                outputs.append(future.result())
            except Exception as e:
                print(f"Chunk failed: {chunk['chunk_id']}, error={e}")

    return combine_chunk_predictions(outputs, len(X))


# ------------------------------------------------------------
# Benchmarking
# ------------------------------------------------------------

def evaluate_predictions(y_true, y_pred):
    """
    Return accuracy and confusion matrix.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def benchmark_inference(model, X_test, y_test):
    """
    Benchmark serial, threaded, and process-based inference.
    """

    rows = []

    print("\nRunning serial baseline...")

    serial_timing = time_function(
        lambda: serial_predict(model, X_test),
        repeats=REPEATS,
    )

    serial_pred = serial_timing["result"]
    serial_time = serial_timing["mean"]
    serial_acc = accuracy_score(y_test, serial_pred)

    rows.append({
        "method": "serial",
        "workers": 1,
        "chunk_size": 0,
        "mean_time": serial_time,
        "speedup": 1.0,
        "samples_per_sec": len(X_test) / serial_time,
        "accuracy": serial_acc,
        "correct": True,
    })

    print(f"Serial time: {serial_time:.6f} seconds")
    print(f"Serial accuracy: {serial_acc:.4f}")

    # Threaded experiments
    for workers in WORKER_COUNTS:
        for chunk_size in CHUNK_SIZES:
            print(f"\nRunning threaded: workers={workers}, chunk_size={chunk_size}")

            timing = time_function(
                lambda w=workers, c=chunk_size: threaded_predict(
                    model,
                    X_test,
                    workers=w,
                    chunk_size=c,
                ),
                repeats=REPEATS,
            )

            pred = timing["result"]
            elapsed = timing["mean"]

            correct = np.array_equal(pred, serial_pred)
            acc = accuracy_score(y_test, pred)

            rows.append({
                "method": "threaded",
                "workers": workers,
                "chunk_size": chunk_size,
                "mean_time": elapsed,
                "speedup": serial_time / elapsed,
                "samples_per_sec": len(X_test) / elapsed,
                "accuracy": acc,
                "correct": correct,
            })

            print(
                f"time={elapsed:.6f}s | "
                f"speedup={serial_time / elapsed:.2f}x | "
                f"samples/sec={len(X_test) / elapsed:.1f} | "
                f"accuracy={acc:.4f} | "
                f"correct={correct}"
            )

    # Process experiments
    for workers in WORKER_COUNTS:
        for chunk_size in CHUNK_SIZES:
            print(f"\nRunning process: workers={workers}, chunk_size={chunk_size}")

            timing = time_function(
                lambda w=workers, c=chunk_size: process_predict(
                    model,
                    X_test,
                    workers=w,
                    chunk_size=c,
                ),
                repeats=REPEATS,
            )

            pred = timing["result"]
            elapsed = timing["mean"]

            correct = np.array_equal(pred, serial_pred)
            acc = accuracy_score(y_test, pred)

            rows.append({
                "method": "process",
                "workers": workers,
                "chunk_size": chunk_size,
                "mean_time": elapsed,
                "speedup": serial_time / elapsed,
                "samples_per_sec": len(X_test) / elapsed,
                "accuracy": acc,
                "correct": correct,
            })

            print(
                f"time={elapsed:.6f}s | "
                f"speedup={serial_time / elapsed:.2f}x | "
                f"samples/sec={len(X_test) / elapsed:.1f} | "
                f"accuracy={acc:.4f} | "
                f"correct={correct}"
            )

    return rows, serial_pred


# ------------------------------------------------------------
# Reporting
# ------------------------------------------------------------

def print_results_table(rows):
    """
    Print benchmark results.
    """

    print()
    print("=" * 112)
    print(
        f"{'Method':12s} "
        f"{'Workers':>8s} "
        f"{'Chunk':>8s} "
        f"{'Mean Time':>12s} "
        f"{'Speedup':>10s} "
        f"{'Samples/sec':>14s} "
        f"{'Accuracy':>10s} "
        f"{'Correct':>10s}"
    )
    print("-" * 112)

    for row in rows:
        print(
            f"{row['method']:12s} "
            f"{row['workers']:8d} "
            f"{row['chunk_size']:8d} "
            f"{row['mean_time']:12.6f} "
            f"{row['speedup']:10.2f} "
            f"{row['samples_per_sec']:14.1f} "
            f"{row['accuracy']:10.4f} "
            f"{str(row['correct']):>10s}"
        )

    print("=" * 112)


def save_results_csv(rows, filename="results.csv"):
    """
    Save benchmark results to CSV.
    """

    fieldnames = [
        "method",
        "workers",
        "chunk_size",
        "mean_time",
        "speedup",
        "samples_per_sec",
        "accuracy",
        "correct",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {filename}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("Day 4 Lab: Parallel ML Inference and Evaluation")

    X_train, X_test, y_train, y_test = make_dataset()

    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    rows, serial_pred = benchmark_inference(
        model,
        X_test,
        y_test,
    )

    print_results_table(rows)
    save_results_csv(rows)

    print("\nReflection Questions:")
    print("1. Which method was fastest: serial, threaded, or process?")
    print("2. Which worker count and chunk size worked best?")
    print("3. Did parallel predictions exactly match the serial baseline?")
    print("4. Did accuracy change? Why or why not?")
    print("5. Why might process-based inference be slower for this small model?")
    print("6. How would results change with a larger model or larger dataset?")
    print("7. How could this pattern support your final project?")


if __name__ == "__main__":
    main()