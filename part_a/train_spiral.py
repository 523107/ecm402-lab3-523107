"""Problem 3-4: train a NumPy MLP on a 3-class spiral dataset."""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from layers import Network
from optimizers import SGD, RMSProp, Adam


def make_spiral(n_per_class=150, noise=0.15, seed=0):
    """Create a 3-class spiral dataset."""
    rng = np.random.default_rng(seed)

    X = np.zeros((n_per_class * 3, 2))
    y = np.zeros(n_per_class * 3, dtype=int)

    for k in range(3):
        ix = range(k * n_per_class, (k + 1) * n_per_class)

        r = np.linspace(0.1, 1.0, n_per_class)
        t = (
            np.linspace(k * 4.0, (k + 1) * 4.0, n_per_class)
            + rng.normal(0, noise, n_per_class)
        )

        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = k

    return X, y


def accuracy(net, X, y):
    return np.mean(net.predict(X) == y)


def train(net, optimizer, X, y, epochs=1000):
    losses = []
    accuracies = []

    for epoch in range(epochs):
        loss = net.forward(X, y)
        net.backward()

        for layer in net.dense:
            optimizer.update(layer)

        losses.append(loss)
        accuracies.append(accuracy(net, X, y))

    return losses, accuracies


def plot_boundary(net, X, y, title, filename):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = net.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, pred, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=20)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main():
    X, y = make_spiral(
        n_per_class=150,
        noise=0.15,
        seed=0
    )

    optimizers = {
        "SGD": SGD(lr=0.1, beta=0.9),
        "RMSProp": RMSProp(lr=0.01),
        "Adam": Adam(lr=0.01),
    }

    results = {}

    for name, optimizer in optimizers.items():
        print(f"\n===== {name} =====")

        net = Network([2, 64, 64, 3], seed=0)

        start = time.perf_counter()

        losses, accuracies = train(
            net,
            optimizer,
            X,
            y,
            epochs=1000
        )

        elapsed = time.perf_counter() - start

        results[name] = {
            "net": net,
            "losses": losses,
            "accuracies": accuracies,
            "time": elapsed,
        }

        print(f"Final loss     : {losses[-1]:.6f}")
        print(f"Final accuracy : {accuracies[-1]:.4f}")
        print(f"Training time  : {elapsed:.3f} s")

    # ------------------------------------------------------------
    # Loss comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    for name, result in results.items():
        plt.plot(result["losses"], label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Optimizer comparison: training loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/optimizer_loss.png",
        dpi=200
    )

    plt.close()

    # ------------------------------------------------------------
    # Accuracy comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    for name, result in results.items():
        plt.plot(result["accuracies"], label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Optimizer comparison: training accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/optimizer_accuracy.png",
        dpi=200
    )

    plt.close()

    # ------------------------------------------------------------
    # Decision boundary for each optimizer
    # ------------------------------------------------------------
    for name, result in results.items():
        plot_boundary(
            result["net"],
            X,
            y,
            f"Decision boundary - {name}",
            f"results/figures/boundary_{name.lower()}.png"
        )

    # ------------------------------------------------------------
    # Save numerical summary
    # ------------------------------------------------------------
    with open("results/tables/optimizer_summary.csv", "w") as f:
        f.write("optimizer,final_loss,final_accuracy,time_seconds\n")

        for name, result in results.items():
            f.write(
                f"{name},"
                f"{result['losses'][-1]:.8f},"
                f"{result['accuracies'][-1]:.8f},"
                f"{result['time']:.6f}\n"
            )

    print("\nAll optimizer experiments completed.")


if __name__ == "__main__":
    main()