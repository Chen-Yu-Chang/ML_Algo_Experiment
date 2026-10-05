"""
EP1: Linear Regression
======================
Linear regression from scratch, using a toy "house size -> house price" dataset.

    Part 0  Generate data + train/test split
    Part 1  From scratch: gradient descent
    Part 2  From scratch: closed-form solution (normal equation)
    Part 3  scikit-learn in three lines
    Part 4  Compare all three on the test set + plot

Usage:
    python linear_regression.py              # default settings
    python linear_regression.py --lr 1.05    # what happens if the learning rate is too big?
    python linear_regression.py --no-scale   # what happens without feature scaling?
    python linear_regression.py --no-show    # save the plot without opening a window

Requires: numpy, matplotlib, scikit-learn (see requirements.txt at the repo root)

Units: size is in "ping" (a Taiwanese unit, about 3.3 m^2),
       price is in units of 10,000 NTD.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------
# Part 0: Generate data
# ---------------------------------------------------------------
# We play god: first we decide the real rule,
#     price = 52 * size + 300 + noise
# then we pretend we don't know 52 and 300 and see if the model can find them.
TRUE_W = 52.0      # price per ping
TRUE_B = 300.0     # base price
NOISE_STD = 150.0  # houses of the same size still differ (location, age, floor...)


def make_data(n=100, seed=42):
    rng = np.random.default_rng(seed)        # fixed seed -> everyone gets the same numbers
    x = rng.uniform(10, 60, size=n)          # size: 10 to 60 ping
    noise = rng.normal(0, NOISE_STD, size=n)
    y = TRUE_W * x + TRUE_B + noise          # price
    return x, y


def train_test_split(x, y, test_ratio=0.2, seed=0):
    """From EP0: never grade the model on data it studied. Lock 20% away as the test set."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_test = int(len(x) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def mse(y_true, y_pred):
    """Mean squared error: square each error, then average. Lower is better."""
    return np.mean((y_pred - y_true) ** 2)


def r2_score(y_true, y_pred):
    """R^2: how much better than always guessing the mean. 1 = perfect, 0 = no better than the mean."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


# ---------------------------------------------------------------
# Part 1: Gradient descent (from scratch)
# ---------------------------------------------------------------
class LinearRegressionGD:
    """
    Model:    y_hat = w * x + b
    Loss:     L = (1/n) * sum((y_hat - y)^2)
    Gradient: dL/dw = (2/n) * sum((y_hat - y) * x)
              dL/db = (2/n) * sum(y_hat - y)
    Update:   w <- w - lr * dL/dw
              b <- b - lr * dL/db
    """

    def __init__(self, lr=0.1, epochs=200, scale=True):
        self.lr = lr
        self.epochs = epochs
        self.scale = scale
        self.loss_history = []
        self.diverged = False

    def fit(self, x, y):
        # --- Feature scaling (standardization) ---
        # Size ranges from about 10 to 60, so x^2 is in the thousands.
        # Without scaling, the slope in the w direction is ~1000x steeper than in b.
        # A learning rate big enough to move b makes w explode;
        # one small enough to keep w stable barely moves b.
        # After scaling x to mean 0 / std 1, both directions have similar slopes.
        if self.scale:
            self.x_mean, self.x_std = x.mean(), x.std()
        else:
            self.x_mean, self.x_std = 0.0, 1.0
        xs = (x - self.x_mean) / self.x_std

        # --- Initialize: start both knobs at 0 ---
        w, b = 0.0, 0.0
        n = len(xs)

        for epoch in range(self.epochs):
            # 1. Forward: predict with the current w, b
            y_hat = w * xs + b

            # 2. Loss (recorded only for plotting)
            error = y_hat - y
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            # Loss grew to a million times its starting value -> it diverged, stop early
            if not np.isfinite(loss) or loss > 1e6 * self.loss_history[0]:
                print(f"  [warning] loss exploded at step {epoch} (learning rate too big?)")
                self.diverged = True
                break

            # 3. Gradient (the slope under your feet)
            dw = (2 / n) * np.sum(error * xs)
            db = (2 / n) * np.sum(error)

            # 4. Take one step downhill
            w -= self.lr * dw
            b -= self.lr * db

        # --- Convert the scaled w, b back to the original units ---
        # y = w * (x - mean) / std + b
        #   = (w / std) * x + (b - w * mean / std)
        self.w = w / self.x_std
        self.b = b - w * self.x_mean / self.x_std
        return self

    def predict(self, x):
        return self.w * x + self.b


# ---------------------------------------------------------------
# Part 2: Closed-form solution (normal equation)
# ---------------------------------------------------------------
def fit_normal_equation(x, y):
    """
    Solve for the loss-minimizing w, b in one step:
        theta = (X^T X)^-1 X^T y
    Each row of X is [1, x]; the leading 1 is what lets us learn b.
    """
    X = np.column_stack([np.ones_like(x), x])
    # solve() instead of inv(): same math, but numerically more stable
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    b, w = theta
    return w, b


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EP1: Linear Regression")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default 0.1)")
    parser.add_argument("--epochs", type=int, default=200, help="gradient descent steps (default 200)")
    parser.add_argument("--no-scale", action="store_true", help="turn off feature scaling")
    parser.add_argument("--no-show", action="store_true", help="don't open the plot window")
    args = parser.parse_args()

    # Part 0
    x, y = make_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(f"{len(x_train)} training samples, {len(x_test)} test samples")
    print(f"True answer: w = {TRUE_W}, b = {TRUE_B}\n")

    # Part 1
    gd = LinearRegressionGD(lr=args.lr, epochs=args.epochs, scale=not args.no_scale)
    gd.fit(x_train, y_train)

    # Part 2
    ne_w, ne_b = fit_normal_equation(x_train, y_train)

    # Part 3: scikit-learn, really just three lines
    sk = LinearRegression()
    sk.fit(x_train.reshape(-1, 1), y_train)   # sklearn wants 2D input: (n_samples, n_features)
    sk_w, sk_b = sk.coef_[0], sk.intercept_

    # Part 4: compare on the unseen test set
    results = [
        ("Gradient descent", gd.w, gd.b, gd.predict(x_test)),
        ("Normal equation", ne_w, ne_b, ne_w * x_test + ne_b),
        ("scikit-learn", sk_w, sk_b, sk.predict(x_test.reshape(-1, 1))),
    ]
    print(f"{'Method':<20}{'w':>10}{'b':>10}{'Test MSE':>12}{'Test R2':>10}")
    print("-" * 62)
    for name, w, b, pred in results:
        if name == "Gradient descent" and gd.diverged:
            print(f"{name:<20}  diverged, no result")
            continue
        print(f"{name:<20}{w:>10.2f}{b:>10.2f}{mse(y_test, pred):>12.1f}{r2_score(y_test, pred):>10.3f}")

    house = 30
    if not gd.diverged:
        print(f"\nGradient descent model's guess for a {house}-ping house: {gd.predict(house):,.0f}")

    # --- Plot: data + fitted line on the left, loss curve on the right ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.scatter(x_train, y_train, s=18, alpha=0.6, label="train")
    ax1.scatter(x_test, y_test, s=30, marker="x", label="test")
    xs = np.linspace(10, 60, 100)
    if not gd.diverged:
        ax1.plot(xs, gd.w * xs + gd.b, linewidth=2, label=f"GD: y = {gd.w:.1f}x + {gd.b:.0f}")
    ax1.plot(xs, TRUE_W * xs + TRUE_B, "--", linewidth=1, label="true rule")
    ax1.set_xlabel("Size (ping)")
    ax1.set_ylabel("Price (10k NTD)")
    ax1.set_title("Linear Regression fit")
    ax1.legend()

    ax2.plot(gd.loss_history)
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE loss (log scale)")
    ax2.set_title(f"Gradient descent (lr = {args.lr})")

    fig.tight_layout()
    fig.savefig("ep01_result.png", dpi=120)
    print("\nPlot saved to ep01_result.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
