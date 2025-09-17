# Lab 8 – A4: Learning-rate sweep with fixed initialization (AND gate)
# Author: S. Udhaya Sankari | Roll: BL.EN.U4CSE23150

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- output folder ----------
U_OUTDIR = r"C:\Users\Udhaya\sem5_ML\lab8_output_figures"
os.makedirs(U_OUTDIR, exist_ok=True)

# ---------- data (AND) ----------
U_X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=float)
U_y = np.array([0.,0.,0.,1.], dtype=float)

# ---------- perceptron pieces ----------
def U_step(z):  # binary step
    return 1.0 if z >= 0 else 0.0

def U_epoch_sse(U_W, X, y, lr):
    """One full epoch with perceptron-style update; returns (new_W, epoch_SSE)."""
    E = 0.0
    for xi, t in zip(X, y):
        xb = np.append(xi, 1.0)          # add bias input
        z = float(xb @ U_W)
        yhat = U_step(z)
        err = t - yhat
        U_W = U_W + lr * err * xb        # perceptron update
        E += 0.5 * (err**2)
    return U_W, E

def U_train_fixed_init(init_W, lr, max_epochs=1000, tol=0.002):
    """Train starting from the SAME init_W for a given lr; return epochs to converge."""
    W = init_W.copy()
    hist = []
    for ep in range(1, max_epochs+1):
        W, E = U_epoch_sse(W, U_X, U_y, lr)
        hist.append(E)
        if E <= tol:
            return ep, W, hist
    return max_epochs, W, hist  # did not meet tol

# ---------- choose the fixed initialization ----------
# Use the *same* initialization that you used in A1.
# Option 1: derive once from a RNG and then freeze it:
rng = np.random.default_rng(7)
U_init_W = rng.normal(0, 1, size=(U_X.shape[1]+1,)) / np.sqrt(U_X.shape[1])
# Option 2 (if your instructor wants A2’s starting point), uncomment:
# U_init_W = np.array([0.2, -0.75, 10.0], dtype=float)

# ---------- LR sweep ----------
U_lrs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
U_results = []

print("[Lab 8 – A4 | LR Sweep | Fixed Init]")
print(f"Fixed init W = [{U_init_W[0]:.4f}, {U_init_W[1]:.4f}, {U_init_W[2]:.4f}]")

for eta in U_lrs:
    epochs, Wfinal, hist = U_train_fixed_init(U_init_W, lr=eta)
    U_results.append((eta, epochs, hist[-1], Wfinal))
    print(f"eta={eta:.1f}  -> epochs={epochs:4d}, final_SSE={hist[-1]:.6f}, "
          f"W=[{Wfinal[0]:.4f}, {Wfinal[1]:.4f}, {Wfinal[2]:.4f}]")

# ---------- plot: epochs to converge vs LR ----------
plt.figure()
plt.plot([r[0] for r in U_results], [r[1] for r in U_results], marker='o')
plt.xlabel("Learning rate (η)")
plt.ylabel("Epochs to Converge (SSE ≤ 0.002; max 1000)")
plt.title("A4: Convergence Speed vs Learning Rate (fixed initialization)")
plt.grid(True)
save_path = os.path.join(U_OUTDIR, "U_A4_epochs_vs_lr.png")
plt.tight_layout(); plt.savefig(save_path, dpi=150)
print(f"\nSaved plot: {save_path}")
