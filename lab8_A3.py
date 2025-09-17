# u_lab8_A3_full.py
# Lab 8 – A3: Compare activations on AND gate (perceptron vs delta updates)
# Author: S. Udhaya Sankari | Roll: BL.EN.U4CSE23150

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# ---------------------------- Banner ----------------------------
def U_banner():
    print("[Lab 8 – A3 | Activation Comparison | S. Udhaya Sankari | BL.EN.U4CSE23150]")

# ---------------------------- Data ------------------------------
def U_and_truth():
    U_X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=float)
    U_y = np.array([0.,0.,0.,1.], dtype=float)
    return U_X, U_y

# ------------------------ Activations ---------------------------
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

# ---------------------- Training Core ---------------------------
def U_train_single(X, y, act="bipolar", mode="perceptron", lr=0.2,
                   max_epochs=1000, seed=7, tol=0.002):
    """
    act: 'bipolar' | 'sigmoid' | 'relu'
    mode: 'perceptron' (discrete class error) OR 'delta' (SSE gradient)
    Returns: epochs_to_conv, final_weights, error_history(list)
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, size=(X.shape[1]+1,)) / np.sqrt(X.shape[1])  # [w1,w2,b]
    hist = []
    for ep in range(1, max_epochs+1):
        E = 0.0
        for xi, target in zip(X, y):
            xb = np.append(xi, 1.0)
            z = float(xb @ W)

            if act == "bipolar":
                a = U_bipolar(z)
                yhat = 1.0 if a > 0 else 0.0
                err = target - yhat
                # step has no gradient; use perceptron-style update
                W += lr * err * xb
                E += 0.5 * (err**2)

            elif act == "sigmoid":
                a = U_sigmoid(z)
                if mode == "delta":
                    err = target - a
                    grad = -err * U_sigmoid_deriv(a) * xb
                    W -= lr * grad
                else:
                    yhat = 1.0 if a >= 0.5 else 0.0
                    err = target - yhat
                    W += lr * err * xb
                E += 0.5 * (err**2)

            elif act == "relu":
                a = U_relu(z)
                if mode == "delta":
                    err = target - a
                    grad = -err * U_relu_deriv(z) * xb
                    W -= lr * grad
                else:
                    yhat = 1.0 if a >= 0.5 else 0.0
                    err = target - yhat
                    W += lr * err * xb
                E += 0.5 * (err**2)

        hist.append(E)
        if E <= tol:
            return ep, W, hist
    return max_epochs, W, hist

# ------------------ Runner & Pretty Printing --------------------
def U_run_all(lr=0.2, seed=7):
    X, y = U_and_truth()
    configs = [
        ("bipolar", "perceptron"),
        ("sigmoid", "perceptron"),
        ("sigmoid", "delta"),
        ("relu",    "perceptron"),
        ("relu",    "delta"),
    ]
    results = OrderedDict()
    for act, mode in configs:
        ep, W, hist = U_train_single(X, y, act=act, mode=mode, lr=lr, seed=seed)
        results[(act, mode)] = (ep, W, hist)
    return results

def U_print_table(results):
    print("\n=== Convergence Summary (AND gate) ===")
    print(f"{'Activation':<12} {'Update':<11} {'Epochs':>8} {'Final_SSE':>12}   Weights[w1,w2,b]")
    for (act, mode), (ep, W, hist) in results.items():
        print(f"{act:<12} {mode:<11} {ep:>8} {hist[-1]:>12.6f}   [{W[0]:.4f}, {W[1]:.4f}, {W[2]:.4f}]")

def U_plot_histories(results):
    # perceptron-style
    plt.figure()
    for act in ["bipolar","sigmoid","relu"]:
        key = (act, "perceptron")
        if key in results:
            hist = results[key][2]
            plt.plot(range(1,len(hist)+1), hist, label=act)
    plt.xlabel("Epochs"); plt.ylabel("Sum-Square Error")
    plt.title("Error vs Epochs (perceptron updates)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("U_A3_perceptron_updates.png", dpi=150)

    # delta-style
    plt.figure()
    for act in ["sigmoid","relu"]:
        key = (act, "delta")
        if key in results:
            hist = results[key][2]
            plt.plot(range(1,len(hist)+1), hist, label=f"{act} (delta)")
    plt.xlabel("Epochs"); plt.ylabel("Sum-Square Error")
    plt.title("Error vs Epochs (delta updates)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("U_A3_delta_updates.png", dpi=150)

# ------------------------------ Main ---------------------------
if __name__ == "__main__":
    U_banner()
    U_results = U_run_all(lr=0.2, seed=7)
    U_print_table(U_results)
    U_plot_histories(U_results)
    print("\nSaved plots: U_A3_perceptron_updates.png, U_A3_delta_updates.png")
