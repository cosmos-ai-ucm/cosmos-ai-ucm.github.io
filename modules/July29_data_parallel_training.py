"""
Day 6 Lab: Data-Parallel Training and Gradient Averaging

Run:

    python July29_data_parallel_training.py

This lab implements:
1. Serial logistic regression training
2. Simulated data-parallel training with gradient averaging
3. Weight averaging / federated-style training
4. Benchmarking across world sizes
5. Loss/accuracy comparison
6. Approximate communication cost

This is a simulation, not real torch.distributed/DDP.
The goal is to understand the algorithmic ideas before using real frameworks.
"""

import time
import csv
import math
import statistics
import numpy as np


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RANDOM_SEED = 42
N_SAMPLES = 6000
N_FEATURES = 30
TRAIN_FRAC = 0.8

EPOCHS = 80
LEARNING_RATE = 0.5

WORLD_SIZES = [1, 2, 4, 8]
REPEATS = 3


# ------------------------------------------------------------
# Dataset generation
# ------------------------------------------------------------

def make_classification_data(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    seed=RANDOM_SEED,
):
    """
    Create a synthetic binary classification dataset.

    We generate a true weight vector and labels from a linear model.
    """

    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    true_b = 0.25

    logits = X @ true_w + true_b
    probs = sigmoid(logits)

    y = (probs >= 0.5).astype(np.float64)

    # Shuffle data
    idx = rng.permutation(n_samples)

    return X[idx], y[idx]


def train_test_split(X, y, train_frac=TRAIN_FRAC):
    """
    Simple train/test split.
    """

    n_train = int(len(X) * train_frac)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# Logistic regression utilities
# ------------------------------------------------------------

def sigmoid(z):
    """
    Numerically stable sigmoid.
    """

    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(params, X):
    """
    Compute predicted probability.
    """

    w = params["w"]
    b = params["b"]

    return sigmoid(X @ w + b)


def predict(params, X):
    """
    Convert probabilities into binary predictions.
    """

    return (predict_proba(params, X) >= 0.5).astype(np.float64)


def binary_cross_entropy(y_true, y_prob):
    """
    Binary cross-entropy loss.
    """

    eps = 1e-8

    return -np.mean(
        y_true * np.log(y_prob + eps)
        + (1.0 - y_true) * np.log(1.0 - y_prob + eps)
    )


def loss_and_grad(params, X, y):
    """
    Compute logistic-regression loss and gradients.

    Returns
    -------
    loss : float
    grads : dict
        Gradients for w and b.
    """

    w = params["w"]
    b = params["b"]

    probs = sigmoid(X @ w + b)
    loss = binary_cross_entropy(y, probs)

    error = probs - y

    grad_w = X.T @ error / len(X)
    grad_b = float(np.mean(error))

    grads = {
        "w": grad_w,
        "b": grad_b,
    }

    return loss, grads


def init_params(n_features):
    """
    Initialize model parameters.
    """

    return {
        "w": np.zeros(n_features, dtype=np.float64),
        "b": 0.0,
    }


def copy_params(params):
    """
    Deep-copy parameter dictionary.
    """

    return {
        "w": params["w"].copy(),
        "b": float(params["b"]),
    }


def apply_gradients(params, grads, lr):
    """
    Apply SGD update.
    """

    params["w"] -= lr * grads["w"]
    params["b"] -= lr * grads["b"]

    return params


def accuracy(params, X, y):
    """
    Classification accuracy.
    """

    y_pred = predict(params, X)
    return float(np.mean(y_pred == y))


# ------------------------------------------------------------
# Data sharding
# ------------------------------------------------------------

def shard_data(X, y, world_size):
    """
    Split data into world_size shards using round-robin slicing.

    Rank r gets:
        X[r::world_size]
    """

    shards = []

    for rank in range(world_size):
        X_shard = X[rank::world_size]
        y_shard = y[rank::world_size]

        shards.append({
            "rank": rank,
            "X": X_shard,
            "y": y_shard,
        })

    return shards


# ------------------------------------------------------------
# Simulated all-reduce operations
# ------------------------------------------------------------

def average_gradients(local_grads):
    """
    Simulate all-reduce average for gradients.

    Every worker contributes a gradient.
    The result is the average gradient.

    In real distributed training, this would be an all-reduce.
    """

    avg_w = np.mean(
        [g["w"] for g in local_grads],
        axis=0,
    )

    avg_b = float(
        np.mean([g["b"] for g in local_grads])
    )

    return {
        "w": avg_w,
        "b": avg_b,
    }


def average_params(local_params):
    """
    Average model weights.

    This is used for the optional weight-averaging / federated-style
    experiment.
    """

    avg_w = np.mean(
        [p["w"] for p in local_params],
        axis=0,
    )

    avg_b = float(
        np.mean([p["b"] for p in local_params])
    )

    return {
        "w": avg_w,
        "b": avg_b,
    }


def count_parameters(params):
    """
    Count number of scalar parameters.
    """

    return params["w"].size + 1


# ------------------------------------------------------------
# Serial training
# ------------------------------------------------------------

def train_serial(
    X_train,
    y_train,
    X_test,
    y_test,
    lr=LEARNING_RATE,
    epochs=EPOCHS,
):
    """
    Train logistic regression on the full training dataset.
    """

    params = init_params(X_train.shape[1])

    history = []

    start = time.perf_counter()

    for epoch in range(epochs):
        train_loss, grads = loss_and_grad(
            params,
            X_train,
            y_train,
        )

        apply_gradients(params, grads, lr)

        train_acc = accuracy(params, X_train, y_train)
        test_acc = accuracy(params, X_test, y_test)

        history.append({
            "method": "serial",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "communication_rounds": 0,
            "communication_scalars": 0,
        })

    elapsed = time.perf_counter() - start

    return params, history, elapsed


# ------------------------------------------------------------
# Data-parallel training: gradient averaging
# ------------------------------------------------------------

def train_data_parallel_gradient_averaging(
    X_train,
    y_train,
    X_test,
    y_test,
    world_size=4,
    lr=LEARNING_RATE,
    epochs=EPOCHS,
):
    """
    Synchronous data-parallel training.

    Algorithm:
    1. Initialize one global parameter vector.
    2. Split data into shards.
    3. Each rank computes local loss and gradient.
    4. Simulate all-reduce by averaging gradients.
    5. Apply same update to global model.
    """

    params = init_params(X_train.shape[1])
    shards = shard_data(X_train, y_train, world_size)

    history = []

    communication_rounds = 0
    communication_scalars = 0
    n_params = count_parameters(params)

    start = time.perf_counter()

    for epoch in range(epochs):
        local_losses = []
        local_grads = []

        for shard in shards:
            loss, grads = loss_and_grad(
                params,
                shard["X"],
                shard["y"],
            )

            local_losses.append(loss)
            local_grads.append(grads)

        # Simulated all-reduce average
        avg_grads = average_gradients(local_grads)

        communication_rounds += 1

        # Approximate communication volume:
        # each worker contributes one gradient vector per synchronization.
        communication_scalars += world_size * n_params

        apply_gradients(params, avg_grads, lr)

        train_loss = float(np.mean(local_losses))
        train_acc = accuracy(params, X_train, y_train)
        test_acc = accuracy(params, X_test, y_test)

        history.append({
            "method": f"grad_avg_world_{world_size}",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "communication_rounds": communication_rounds,
            "communication_scalars": communication_scalars,
        })

    elapsed = time.perf_counter() - start

    return params, history, elapsed


# ------------------------------------------------------------
# Optional: Weight averaging / federated-style training
# ------------------------------------------------------------

def local_train_steps(params, X, y, lr, local_steps):
    """
    Train a local model copy for several local steps.
    """

    local_params = copy_params(params)

    for _ in range(local_steps):
        _, grads = loss_and_grad(local_params, X, y)
        apply_gradients(local_params, grads, lr)

    return local_params


def train_weight_averaging(
    X_train,
    y_train,
    X_test,
    y_test,
    world_size=4,
    lr=LEARNING_RATE,
    rounds=20,
    local_steps=5,
):
    """
    Weight averaging simulation.

    This resembles federated averaging:
    1. Start with global params.
    2. Send params to workers.
    3. Workers train locally for several steps.
    4. Average local weights.
    5. Repeat.

    This is not the same as DDP gradient averaging.
    """

    params = init_params(X_train.shape[1])
    shards = shard_data(X_train, y_train, world_size)

    history = []

    communication_rounds = 0
    communication_scalars = 0
    n_params = count_parameters(params)

    start = time.perf_counter()

    for round_id in range(rounds):
        local_params = []

        for shard in shards:
            trained = local_train_steps(
                params,
                shard["X"],
                shard["y"],
                lr,
                local_steps,
            )

            local_params.append(trained)

        params = average_params(local_params)

        communication_rounds += 1

        # Each worker sends local parameters back.
        communication_scalars += world_size * n_params

        y_prob = predict_proba(params, X_train)
        train_loss = binary_cross_entropy(y_train, y_prob)

        train_acc = accuracy(params, X_train, y_train)
        test_acc = accuracy(params, X_test, y_test)

        history.append({
            "method": f"weight_avg_world_{world_size}_steps_{local_steps}",
            "epoch": round_id,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "communication_rounds": communication_rounds,
            "communication_scalars": communication_scalars,
        })

    elapsed = time.perf_counter() - start

    return params, history, elapsed


# ------------------------------------------------------------
# Benchmarking
# ------------------------------------------------------------

def summarize_run(method, params, history, elapsed, X_test, y_test):
    """
    Create a compact summary row.
    """

    final = history[-1]

    return {
        "method": method,
        "time": elapsed,
        "final_train_loss": final["train_loss"],
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "communication_rounds": final["communication_rounds"],
        "communication_scalars": final["communication_scalars"],
        "num_parameters": count_parameters(params),
    }


def run_experiments():
    """
    Run serial, gradient averaging, and optional weight averaging.
    """

    X, y = make_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("Features:", X_train.shape[1])

    all_histories = []
    summary_rows = []

    # Serial baseline
    print("\nRunning serial baseline...")

    serial_params, serial_history, serial_time = train_serial(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    summary_rows.append(
        summarize_run(
            "serial",
            serial_params,
            serial_history,
            serial_time,
            X_test,
            y_test,
        )
    )

    all_histories.extend(serial_history)

    print(
        f"serial | time={serial_time:.4f}s | "
        f"test_acc={serial_history[-1]['test_acc']:.4f}"
    )

    # Data-parallel gradient averaging
    for world_size in WORLD_SIZES:
        print(f"\nRunning gradient averaging with world_size={world_size}...")

        params, history, elapsed = train_data_parallel_gradient_averaging(
            X_train,
            y_train,
            X_test,
            y_test,
            world_size=world_size,
        )

        method = f"grad_avg_world_{world_size}"

        summary_rows.append(
            summarize_run(
                method,
                params,
                history,
                elapsed,
                X_test,
                y_test,
            )
        )

        all_histories.extend(history)

        print(
            f"{method} | time={elapsed:.4f}s | "
            f"test_acc={history[-1]['test_acc']:.4f} | "
            f"comm_rounds={history[-1]['communication_rounds']}"
        )

    # Optional: weight averaging
    for world_size in [2, 4, 8]:
        for local_steps in [1, 5, 10]:
            print(
                f"\nRunning weight averaging "
                f"world_size={world_size}, local_steps={local_steps}..."
            )

            params, history, elapsed = train_weight_averaging(
                X_train,
                y_train,
                X_test,
                y_test,
                world_size=world_size,
                rounds=20,
                local_steps=local_steps,
            )

            method = f"weight_avg_world_{world_size}_steps_{local_steps}"

            summary_rows.append(
                summarize_run(
                    method,
                    params,
                    history,
                    elapsed,
                    X_test,
                    y_test,
                )
            )

            all_histories.extend(history)

            print(
                f"{method} | time={elapsed:.4f}s | "
                f"test_acc={history[-1]['test_acc']:.4f} | "
                f"comm_rounds={history[-1]['communication_rounds']}"
            )

    return summary_rows, all_histories


# ------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------

def print_summary_table(rows):
    """
    Print compact experiment summary.
    """

    print()
    print("=" * 130)
    print(
        f"{'Method':36s} "
        f"{'Time':>10s} "
        f"{'Loss':>10s} "
        f"{'Train Acc':>10s} "
        f"{'Test Acc':>10s} "
        f"{'Comm Rounds':>12s} "
        f"{'Comm Scalars':>14s}"
    )
    print("-" * 130)

    for r in rows:
        print(
            f"{r['method']:36s} "
            f"{r['time']:10.4f} "
            f"{r['final_train_loss']:10.4f} "
            f"{r['final_train_acc']:10.4f} "
            f"{r['final_test_acc']:10.4f} "
            f"{r['communication_rounds']:12d} "
            f"{r['communication_scalars']:14d}"
        )

    print("=" * 130)


def save_summary_csv(rows, filename="results.csv"):
    """
    Save experiment summary.
    """

    fieldnames = [
        "method",
        "time",
        "final_train_loss",
        "final_train_acc",
        "final_test_acc",
        "communication_rounds",
        "communication_scalars",
        "num_parameters",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary to {filename}")


def save_history_csv(histories, filename="loss_curves.csv"):
    """
    Save per-epoch training history.
    """

    fieldnames = [
        "method",
        "epoch",
        "train_loss",
        "train_acc",
        "test_acc",
        "communication_rounds",
        "communication_scalars",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(histories)

    print(f"Saved loss curves to {filename}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("Day 6 Lab: Data-Parallel Training Simulation")

    rows, histories = run_experiments()

    print_summary_table(rows)

    save_summary_csv(rows)
    save_history_csv(histories)

    print("\nReflection Questions:")
    print("1. Did gradient averaging achieve similar accuracy to serial training?")
    print("2. How did world size affect runtime?")
    print("3. How did world size affect communication_scalars?")
    print("4. How is gradient averaging different from weight averaging?")
    print("5. What happened when local_steps increased in weight averaging?")
    print("6. How does all-reduce connect to PyTorch DDP?")
    print("7. How could this idea support your final project?")


if __name__ == "__main__":
    main()