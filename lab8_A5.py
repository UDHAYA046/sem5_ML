# u_lab8_A5_XOR_all.py
# Lab 8 – A5: XOR with (A1) scratch perceptron, (A2) custom init, (A3) activation comparison
# Author: S. Udhaya Sankari | Roll: BL.EN.U4CSE23150

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

# ---------------------------- Output directory ----------------------------
U_OUTDIR = r"C:\Users\Udhaya\sem5_ML\lab8_output_figures"
os.makedirs(U_OUTDIR, exist_ok=True)

# ---------------------------- Data: XOR gate ------------------------------
# Inputs and labels for XOR
U_X_XOR = np.array([[0., 0.],
                    [0., 1.],
                    [1., 0.],
                    [1., 1.]], dtype=float)
U_y_XOR = np.array([0, 1, 1, 0], dtype=int)

# ---------------------------- Helpers -------------------------------------
def U_banner():
    print("[Lab 8 – A5 | XOR | S. Udhaya Sankari | BL.EN.U4CSE23150]")

def U_step(z):  # binary step (0/1)
    return 1.0 if z >= 0 else 0.0

def U_bipolar(z):  # returns {-1,0,1}
    return 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)

def U_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def U_sigmoid_deriv(a):  # a = sigmoid(z)
    return a * (1.0 - a)

def U_relu(z):
    return z if z > 0 else 0.0

def U_relu_deriv(z):
    return 1.0 if z > 0 else 0.0

# ---------------------------- A1: Scratch perceptron -----------------------
def U_train_perceptron_scratch(X, y, lr=0.2, epochs=2000, seed=7):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, size=(X.shape[1] + 1,)) / np.sqrt(X.shape[1])  # [w1, w2, b]
    for _ in range(epochs):
        for xi, t in zip(X, y):
            xb = np.append(xi, 1.0)
            z = float(xb @ W)
            yhat = U_step(z)
            err = t - yhat
            W += lr * err * xb
    return W

def U_run_A1_XOR():
    print("\n=== A5–A1 (Scratch perceptron on XOR) ===")
    W = U_train_perceptron_scratch(U_X_XOR, U_y_XOR, lr=0.2, epochs=2000, seed=7)
    # predictions
    preds = []
    for xi in U_X_XOR:
        xb = np.append(xi, 1.0)
        preds.append(int(U_step(float(xb @ W))))
    preds = np.array(preds)
    print("Inputs:\n", U_X_XOR)
    print("Targets:     ", U_y_XOR.tolist())
    print("Predictions: ", preds.tolist())
    print(f"Final weights [w1,w2,b]: [{W[0]:.4f}, {W[1]:.4f}, {W[2]:.4f}]")
    ok = np.array_equal(preds, U_y_XOR)
    print("Status:", "✅ perfect" if ok else "❌ cannot learn XOR with a single perceptron")

# ---------------------------- A2: Custom init + error curve ----------------
def U_run_A2_XOR():
    print("\n=== A5–A2 (Custom init, Step activation, XOR) ===")
    # Given init from A2: W0=10 (bias), W1=0.2, W2=-0.75 ; lr = 0.05
    # We'll store as [w1, w2, b]
    W = np.array([0.2, -0.75, 10.0], dtype=float)
    lr = 0.05
    max_epochs = 1000
    tol = 0.002
    err_hist = []

    for ep in range(1, max_epochs + 1):
        E = 0.0
        for xi, t in zip(U_X_XOR, U_y_XOR):
            xb = np.append(xi, 1.0)
            z = float(xb @ W)
            yhat = U_step(z)
            err = t - yhat
            W += lr * err * xb
            E += 0.5 * (err ** 2)
        err_hist.append(E)
        if E <= tol:
            print(f"Converged at epoch {ep}")  # (will not happen for XOR)
            break

    print(f"Final weights after {len(err_hist)} epochs: [{W[0]:.4f}, {W[1]:.4f}, {W[2]:.4f}]")
    print(f"Final SSE: {err_hist[-1]:.6f} (XOR not linearly separable -> will not hit 0)")

    # plot and save
    plt.figure()
    plt.plot(range(1, len(err_hist) + 1), err_hist, marker='o')
    plt.xlabel("Epochs"); plt.ylabel("Sum-Square Error")
    plt.title("A5–A2: XOR (custom init, step activation)")
    plt.grid(True); plt.tight_layout()
    p = os.path.join(U_OUTDIR, "U_A5_XOR_A2_error.png")
    plt.savefig(p, dpi=150)
    print(f"Saved figure: {p}")

# ---------------------------- A3: Activation comparison --------------------
def U_train_single(X, y, act="bipolar", mode="perceptron", lr=0.2,
                   max_epochs=1000, seed=7, tol=0.002):
    """
    act: 'bipolar' | 'sigmoid' | 'relu'
    mode: 'perceptron' (error-driven) | 'delta' (SSE gradient for sigmoid/relu)
    Returns (epochs_to_stop, final_W, error_history)
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, size=(X.shape[1] + 1,)) / np.sqrt(X.shape[1])
    hist = []

    for ep in range(1, max_epochs + 1):
        E = 0.0
        for xi, t in zip(X, y):
            xb = np.append(xi, 1.0)
            z = float(xb @ W)

            if act == "bipolar":
                a = U_bipolar(z)
                yhat = 1.0 if a > 0 else 0.0
                err = t - yhat
                W += lr * err * xb
                E += 0.5 * (err ** 2)

            elif act == "sigmoid":
                a = U_sigmoid(z)
                if mode == "delta":
                    err = t - a
                    grad = -err * U_sigmoid_deriv(a) * xb
                    W -= lr * grad
                else:
                    yhat = 1.0 if a >= 0.5 else 0.0
                    err = t - yhat
                    W += lr * err * xb
                E += 0.5 * (err ** 2)

            elif act == "relu":
                a = U_relu(z)
                if mode == "delta":
                    err = t - a
                    grad = -err * U_relu_deriv(z) * xb
                    W -= lr * grad
                else:
                    yhat = 1.0 if a >= 0.5 else 0.0
                    err = t - yhat
                    W += lr * err * xb
                E += 0.5 * (err ** 2)

        hist.append(E)
        if E <= tol:            # for XOR this condition will NOT be met
            return ep, W, hist
    return max_epochs, W, hist

def U_run_A3_XOR():
    print("\n=== A5–A3 (Activation comparison on XOR) ===")
    X, y = U_X_XOR, U_y_XOR.astype(float)

    results = OrderedDict()
    # perceptron updates for all three activations
    for act in ["bipolar", "sigmoid", "relu"]:
        ep, W, hist = U_train_single(X, y, act=act, mode="perceptron", lr=0.2)
        results[(act, "perceptron")] = (ep, W, hist)

    # delta rule for sigmoid & relu
    for act in ["sigmoid", "relu"]:
        ep, W, hist = U_train_single(X, y, act=act, mode="delta", lr=0.2)
        results[(act, "delta")] = (ep, W, hist)

    # print table
    print("\nConvergence summary (XOR):")
    print(f"{'Activation':<10} {'Update':<11} {'Epochs':>7} {'Final_SSE':>12}   Weights[w1,w2,b]")
    for (act, mode), (ep, W, hist) in results.items():
        print(f"{act:<10} {mode:<11} {ep:>7} {hist[-1]:>12.6f}   "
              f"[{W[0]:.4f}, {W[1]:.4f}, {W[2]:.4f}]")

    # Plot 1: perceptron-style
    plt.figure()
    for act in ["bipolar", "sigmoid", "relu"]:
        k = (act, "perceptron")
        hist = results[k][2]
        plt.plot(range(1, len(hist) + 1), hist, label=act)
    plt.xlabel("Epochs"); plt.ylabel("Sum-Square Error")
    plt.title("A5–A3: XOR (perceptron updates)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    p1 = os.path.join(U_OUTDIR, "U_A5_XOR_A3_perceptron_updates.png")
    plt.savefig(p1, dpi=150)

    # Plot 2: delta-style (sigmoid & relu)
    plt.figure()
    for act in ["sigmoid", "relu"]:
        k = (act, "delta")
        hist = results[k][2]
        plt.plot(range(1, len(hist) + 1), hist, label=f"{act} (delta)")
    plt.xlabel("Epochs"); plt.ylabel("Sum-Square Error")
    plt.title("A5–A3: XOR (delta updates)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    p2 = os.path.join(U_OUTDIR, "U_A5_XOR_A3_delta_updates.png")
    plt.savefig(p2, dpi=150)

    print(f"\nSaved plots:\n  {p1}\n  {p2}")

# ---------------------------- Main ----------------------------------------
if __name__ == "__main__":
    U_banner()
    U_run_A1_XOR()   # scratch perceptron – will show misclassification
    U_run_A2_XOR()   # custom init + error curve – will not converge to 0
    U_run_A3_XOR()   # activation comparison – none will reach SSE<=0.002
