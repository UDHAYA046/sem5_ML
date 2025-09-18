# Lab 8 – A8 | AND via 2-2-1 MLP (Sigmoid + Backprop)
# Author: S. Udhaya Sankari | BL.EN.U4CSE23150

import os, math
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Config --------------------
U_SEED = 42
rng = np.random.default_rng(U_SEED)

U_ALPHA = 0.05                     # learning rate
U_MAX_EPOCHS = 1000
U_TARGET_SSE = 0.002               # stop when SSE <= this
U_OUTPUT_DIR = r"C:\Users\Udhaya\sem5_ML\lab8_output_figures"
os.makedirs(U_OUTPUT_DIR, exist_ok=True)

# -------------------- Data (AND) --------------------
# Inputs (A, B) and targets O1
U_X = np.array([[0., 0.],
                [0., 1.],
                [1., 0.],
                [1., 1.]], dtype=float)
U_T = np.array([[0.],
                [0.],
                [0.],
                [1.]], dtype=float)

# -------------------- Helpers --------------------
def U_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def U_dsigmoid(a):
    # 'a' is already sigmoid(z)
    return a * (1.0 - a)

# -------------------- Network: 2-2-1 --------------------
# We include biases explicitly as separate weight vectors
# Input layer (2) -> Hidden layer (2): V (2x2) and hidden bias b_h (2,)
U_V = rng.normal(0.0, 0.5, size=(2, 2))      # v11 v12 / v21 v22
U_bh = rng.normal(0.0, 0.5, size=(2, ))

# Hidden (2) -> Output (1): W (2x1) and output bias b_o (1,)
U_W = rng.normal(0.0, 0.5, size=(2, 1))      # w1, w2
U_bo = rng.normal(0.0, 0.5, size=(1, ))

def U_forward(x_row):
    """Forward pass for a single example."""
    # hidden pre-activation and activation
    z_h = x_row @ U_V + U_bh        # shape (2,)
    a_h = U_sigmoid(z_h)            # shape (2,)
    # output layer
    z_o = a_h @ U_W + U_bo          # shape (1,)
    a_o = U_sigmoid(z_o)            # shape (1,)
    return a_h, a_o

# -------------------- Training (online SGD) --------------------
U_loss_curve = []
for U_epoch in range(1, U_MAX_EPOCHS + 1):
    # stochastic update over all 4 samples
    U_sse = 0.0
    for i in range(len(U_X)):
        x = U_X[i]                 # shape (2,)
        t = U_T[i]                 # shape (1,)

        # ---- forward
        a_h, y = U_forward(x)      # a_h: (2,), y: (1,)
        # ---- loss
        err = y - t                # (1,)
        U_sse += float(err.T @ err)

        # ---- backprop deltas
        delta_o = err * U_dsigmoid(y)                 # (1,)
        delta_h = (U_W.flatten() * delta_o) * U_dsigmoid(a_h)  # (2,)

        # ---- gradient updates
        # Hidden->Output weights and bias
        U_W -= U_ALPHA * a_h.reshape(2, 1) @ delta_o.reshape(1, 1)
        U_bo -= U_ALPHA * delta_o

        # Input->Hidden weights and biases
        U_V -= U_ALPHA * np.outer(x, delta_h)        # (2x2)
        U_bh -= U_ALPHA * delta_h

    U_loss_curve.append(U_sse)

    if U_sse <= U_TARGET_SSE:
        break

# -------------------- Results --------------------
print("\n=== A8: AND via Backprop (2-2-1, sigmoid) ===")
print(f"Final epoch: {U_epoch}")
print(f"Final SSE  : {U_sse:.6f}")

# Predictions after training
U_preds = []
for i in range(len(U_X)):
    _, y = U_forward(U_X[i])
    U_preds.append(float(y))
U_preds = np.array(U_preds)
U_bin = (U_preds >= 0.5).astype(int)

print("Outputs (sigmoid):", np.round(U_preds, 6))
print("Binarized (>=0.5):", U_bin.tolist())

print("\nFinal weights:")
print("V (input->hidden):")
print(np.round(U_V, 6))
print("b_h:", np.round(U_bh, 6))
print("W (hidden->output):")
print(np.round(U_W, 6))
print("b_o:", np.round(U_bo, 6))

# -------------------- Plot & Save --------------------
plt.figure(figsize=(6, 4))
plt.plot(U_loss_curve)
plt.title("A8 – AND via Backprop: SSE vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Sum-Square Error")
plt.grid(True, linestyle="--", linewidth=0.6)
U_plot_path = os.path.join(U_OUTPUT_DIR, "U_A8_AND_loss.png")
plt.savefig(U_plot_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved plot:", U_plot_path)
