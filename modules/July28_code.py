"""
Day 5 Lab: mini-MPI Collective Simulator

This lab simulates MPI-style collective communication:
1. broadcast
2. scatter
3. gather
4. reduce
5. all-reduce

Then it applies the collectives to:
- distributed word count
- distributed numeric statistics

Important:
- This is not real MPI.
- We are simulating MPI-style ideas using Python multiprocessing.
- Use this to understand communication patterns before Day 6 data-parallel training.
"""

import time
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


# ------------------------------------------------------------
# mini-MPI collective primitives
# ------------------------------------------------------------

def broadcast(value, world_size):
    """
    Simulate MPI_Bcast.

    One root value is copied to every rank.

    Parameters
    ----------
    value : any
        Value to broadcast.
    world_size : int
        Number of simulated ranks.

    Returns
    -------
    list
        One copy of value per rank.
    """

    return [value for _ in range(world_size)]


def scatter(data, world_size):
    """
    Simulate MPI_Scatter.

    Split data into world_size chunks.
    Chunk i goes to rank i.

    Parameters
    ----------
    data : list
        Data to split.
    world_size : int
        Number of ranks.

    Returns
    -------
    list[list]
        List of chunks, one per rank.
    """

    chunks = [[] for _ in range(world_size)]

    for i, item in enumerate(data):
        rank = i % world_size
        chunks[rank].append(item)

    return chunks


def gather(local_results):
    """
    Simulate MPI_Gather.

    In this simplified simulation, worker outputs are already
    returned to the root as a list.

    Parameters
    ----------
    local_results : list
        Local result from each rank.

    Returns
    -------
    list
        Gathered results.
    """

    return local_results


def reduce_sum(values):
    """
    Simulate MPI_Reduce with sum operation.

    Parameters
    ----------
    values : list[number]
        Values from ranks.

    Returns
    -------
    number
        Sum of all values.
    """

    return sum(values)


def all_reduce_sum(values):
    """
    Simulate MPI_Allreduce with sum operation.

    Every rank receives the reduced value.

    Parameters
    ----------
    values : list[number]
        Values from all ranks.

    Returns
    -------
    list[number]
        Same reduced value repeated for every rank.
    """

    total = reduce_sum(values)
    return [total for _ in values]


def reduce_counters(counters):
    """
    Reduce a list of Counter objects by summing counts.

    Parameters
    ----------
    counters : list[Counter]

    Returns
    -------
    Counter
    """

    total = Counter()

    for c in counters:
        total.update(c)

    return total


# ------------------------------------------------------------
# Data generation
# ------------------------------------------------------------

def make_fake_lines(n=50_000):
    """
    Generate synthetic text lines.

    This avoids external dataset dependencies.
    """

    vocab = [
        "ai",
        "data",
        "model",
        "parallel",
        "rank",
        "scatter",
        "gather",
        "reduce",
        "python",
        "training",
        "inference",
        "worker",
    ]

    lines = []

    for i in range(n):
        line = " ".join(vocab[(i + j) % len(vocab)] for j in range(20))
        lines.append(line)

    return lines


# ------------------------------------------------------------
# Local worker functions
# ------------------------------------------------------------

def local_word_count(lines):
    """
    Count words in a local shard.
    """

    counter = Counter()

    for line in lines:
        tokens = line.lower().split()
        counter.update(tokens)

    return counter


def worker_word_count(args):
    """
    Worker function for ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple
        (rank, chunk)

    Returns
    -------
    dict
        Worker result.
    """

    rank, chunk = args

    start = time.perf_counter()
    counter = local_word_count(chunk)
    elapsed = time.perf_counter() - start

    return {
        "rank": rank,
        "num_lines": len(chunk),
        "counter": counter,
        "worker_time": elapsed,
    }


# ------------------------------------------------------------
# Serial baseline
# ------------------------------------------------------------

def serial_word_count(lines):
    """
    Serial baseline for correctness comparison.
    """

    return local_word_count(lines)


# ------------------------------------------------------------
# Distributed word count
# ------------------------------------------------------------

def distributed_word_count(lines, world_size=4):
    """
    Simulate distributed word count.

    Workflow:
    1. Scatter lines to ranks.
    2. Each rank computes local word count.
    3. Gather local counters.
    4. Reduce counters into a global counter.
    """

    chunks = scatter(lines, world_size)

    tasks = [
        (rank, chunk)
        for rank, chunk in enumerate(chunks)
    ]

    with ProcessPoolExecutor(max_workers=world_size) as executor:
        worker_outputs = list(executor.map(worker_word_count, tasks))

    gathered = gather(worker_outputs)

    counters = [
        out["counter"]
        for out in gathered
    ]

    global_counter = reduce_counters(counters)

    return global_counter, gathered


# ------------------------------------------------------------
# Numeric statistics example
# ------------------------------------------------------------

def make_numbers(n=1_000_000):
    """
    Generate synthetic numeric data.
    """

    return list(range(n))


def local_numeric_stats(numbers):
    """
    Compute local numeric statistics.

    Returns partial sum, count, min, and max.
    """

    if not numbers:
        return {
            "sum": 0,
            "count": 0,
            "min": None,
            "max": None,
        }

    return {
        "sum": sum(numbers),
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def worker_numeric_stats(args):
    """
    Worker function for numeric statistics.
    """

    rank, chunk = args

    start = time.perf_counter()
    stats = local_numeric_stats(chunk)
    elapsed = time.perf_counter() - start

    stats["rank"] = rank
    stats["worker_time"] = elapsed

    return stats


def reduce_numeric_stats(partials):
    """
    Reduce partial numeric statistics.
    """

    total_sum = sum(p["sum"] for p in partials)
    total_count = sum(p["count"] for p in partials)

    mins = [
        p["min"]
        for p in partials
        if p["min"] is not None
    ]

    maxs = [
        p["max"]
        for p in partials
        if p["max"] is not None
    ]

    return {
        "sum": total_sum,
        "count": total_count,
        "mean": total_sum / total_count if total_count else None,
        "min": min(mins) if mins else None,
        "max": max(maxs) if maxs else None,
    }


def distributed_numeric_stats(numbers, world_size=4):
    """
    Distributed numeric statistics using scatter/gather/reduce.
    """

    chunks = scatter(numbers, world_size)

    tasks = [
        (rank, chunk)
        for rank, chunk in enumerate(chunks)
    ]

    with ProcessPoolExecutor(max_workers=world_size) as executor:
        partials = list(executor.map(worker_numeric_stats, tasks))

    gathered = gather(partials)
    reduced = reduce_numeric_stats(gathered)

    return reduced, gathered


# ------------------------------------------------------------
# Benchmarking utilities
# ------------------------------------------------------------

def benchmark_word_count(lines, world_sizes):
    """
    Benchmark distributed word count for different world sizes.
    """

    rows = []

    print("\nRunning serial word count baseline...")

    start = time.perf_counter()
    serial_counter = serial_word_count(lines)
    serial_time = time.perf_counter() - start

    rows.append({
        "task": "word_count",
        "world_size": 1,
        "method": "serial",
        "time": serial_time,
        "speedup": 1.0,
        "throughput": len(lines) / serial_time,
        "correct": True,
        "max_worker_time": serial_time,
        "min_worker_time": serial_time,
    })

    print(f"Serial time: {serial_time:.4f} seconds")
    print("Serial top 5:", serial_counter.most_common(5))

    for world_size in world_sizes:
        print(f"\nRunning distributed word count with world_size={world_size}")

        start = time.perf_counter()
        distributed_counter, outputs = distributed_word_count(
            lines,
            world_size=world_size,
        )
        elapsed = time.perf_counter() - start

        worker_times = [
            out["worker_time"]
            for out in outputs
        ]

        chunk_sizes = [
            out["num_lines"]
            for out in outputs
        ]

        correct = distributed_counter == serial_counter
        speedup = serial_time / elapsed
        throughput = len(lines) / elapsed

        rows.append({
            "task": "word_count",
            "world_size": world_size,
            "method": "distributed",
            "time": elapsed,
            "speedup": speedup,
            "throughput": throughput,
            "correct": correct,
            "max_worker_time": max(worker_times),
            "min_worker_time": min(worker_times),
        })

        print(
            f"world_size={world_size} | "
            f"time={elapsed:.4f}s | "
            f"speedup={speedup:.2f}x | "
            f"throughput={throughput:.1f} lines/s | "
            f"correct={correct}"
        )

        print("chunk sizes:", chunk_sizes)
        print("worker times:", [round(t, 4) for t in worker_times])

    return rows


def benchmark_numeric_stats(numbers, world_sizes):
    """
    Benchmark distributed numeric statistics.
    """

    rows = []

    print("\nRunning serial numeric stats baseline...")

    start = time.perf_counter()
    serial_stats = local_numeric_stats(numbers)
    serial_reduced = reduce_numeric_stats([serial_stats])
    serial_time = time.perf_counter() - start

    rows.append({
        "task": "numeric_stats",
        "world_size": 1,
        "method": "serial",
        "time": serial_time,
        "speedup": 1.0,
        "throughput": len(numbers) / serial_time,
        "correct": True,
        "max_worker_time": serial_time,
        "min_worker_time": serial_time,
    })

    print(f"Serial time: {serial_time:.4f} seconds")
    print("Serial stats:", serial_reduced)

    for world_size in world_sizes:
        print(f"\nRunning distributed numeric stats with world_size={world_size}")

        start = time.perf_counter()
        dist_stats, partials = distributed_numeric_stats(
            numbers,
            world_size=world_size,
        )
        elapsed = time.perf_counter() - start

        worker_times = [
            p["worker_time"]
            for p in partials
        ]

        correct = (
            dist_stats["sum"] == serial_reduced["sum"]
            and dist_stats["count"] == serial_reduced["count"]
            and dist_stats["min"] == serial_reduced["min"]
            and dist_stats["max"] == serial_reduced["max"]
            and abs(dist_stats["mean"] - serial_reduced["mean"]) < 1e-9
        )

        speedup = serial_time / elapsed
        throughput = len(numbers) / elapsed

        rows.append({
            "task": "numeric_stats",
            "world_size": world_size,
            "method": "distributed",
            "time": elapsed,
            "speedup": speedup,
            "throughput": throughput,
            "correct": correct,
            "max_worker_time": max(worker_times),
            "min_worker_time": min(worker_times),
        })

        print(
            f"world_size={world_size} | "
            f"time={elapsed:.4f}s | "
            f"speedup={speedup:.2f}x | "
            f"throughput={throughput:.1f} nums/s | "
            f"correct={correct}"
        )

    return rows


def print_results_table(rows):
    """
    Print benchmark rows.
    """

    print()
    print("=" * 115)
    print(
        f"{'Task':16s} "
        f"{'Method':14s} "
        f"{'World':>6s} "
        f"{'Time':>10s} "
        f"{'Speedup':>10s} "
        f"{'Throughput':>14s} "
        f"{'Correct':>10s} "
        f"{'MinW':>10s} "
        f"{'MaxW':>10s}"
    )
    print("-" * 115)

    for r in rows:
        print(
            f"{r['task']:16s} "
            f"{r['method']:14s} "
            f"{r['world_size']:6d} "
            f"{r['time']:10.4f} "
            f"{r['speedup']:10.2f} "
            f"{r['throughput']:14.1f} "
            f"{str(r['correct']):>10s} "
            f"{r['min_worker_time']:10.4f} "
            f"{r['max_worker_time']:10.4f}"
        )

    print("=" * 115)


def save_results_csv(rows, filename="results.csv"):
    """
    Save benchmark rows to CSV.
    """

    fieldnames = [
        "task",
        "method",
        "world_size",
        "time",
        "speedup",
        "throughput",
        "correct",
        "min_worker_time",
        "max_worker_time",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {filename}")


# ------------------------------------------------------------
# Demonstration of simple collectives
# ------------------------------------------------------------

def demo_collectives():
    """
    Demonstrate basic collective operations.
    """

    print("\n=== Collective demo ===")

    world_size = 4

    print("broadcast:", broadcast("model_config_v1", world_size))

    data = list(range(16))
    chunks = scatter(data, world_size)
    print("scatter:", chunks)

    local_values = [
        len(chunk)
        for chunk in chunks
    ]

    print("local values:", local_values)
    print("gather:", gather(local_values))
    print("reduce_sum:", reduce_sum(local_values))
    print("all_reduce_sum:", all_reduce_sum(local_values))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("July28 Lab: mini-MPI Collective Simulator")

    demo_collectives()

    world_sizes = [1, 2, 4, 8]

    # Word-count task
    lines = make_fake_lines(n=50_000)
    word_rows = benchmark_word_count(
        lines,
        world_sizes=world_sizes,
    )

    # Numeric task
    numbers = make_numbers(n=1_000_000)
    numeric_rows = benchmark_numeric_stats(
        numbers,
        world_sizes=world_sizes,
    )

    rows = word_rows + numeric_rows

    print_results_table(rows)
    save_results_csv(rows)

    print("\nReflection Questions:")
    print("1. What is the difference between scatter and broadcast?")
    print("2. What is the difference between gather and reduce?")
    print("3. Did distributed results match serial results?")
    print("4. When did speedup flatten?")
    print("5. Were worker times balanced?")
    print("6. What communication pattern would your final project use?")
    print("7. How does all-reduce connect to distributed training?")


if __name__ == "__main__":
    main()
