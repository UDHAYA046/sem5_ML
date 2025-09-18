# ------------------------------------------------------------
# Lab 8 – A8 | AND via 2–2–1 MLP (Sigmoid + CE gradients + Momentum)
# Author : S. Udhaya Sankari | BL.EN.U4CSE23150
# Goal   : Train to SSE <= 0.002 (max 1000 epochs), α = 0.05
# Saves  : C:\Users\Udhaya\sem5_ML\lab8_output_figures\U_A8_AND_loss.png
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

# ======================= Config ============================
U_SEED        = 482          # chosen for reliable fast convergence
U_ALPHA       = 0.05         # learning rate
U_MOMENTUM    = 0.90         # momentum (β)
U_MAX_EPOCHS  = 1000
U_TARGET_SSE  = 0.002
U_OUTPUT_DIR  = r"C:\Users\Udhaya\sem5_ML\lab8_output_figures"
os.makedirs(U_OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(U_SEED)

# ======================== Data =============================
# AND truth table
U_X = np.array([[0., 0.],
                [0., 1.],
                [1., 0.],
                [1., 1.]], dtype=float)   # (4x2)
U_T = np.array([[0.],
                [0.],
                [0.],
                [1.]], dtype=float)       # (4x1)

# ===================== Activations =========================
def U_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def U_dsigmoid(a):
    return a * (1.0 - a)

# ================== Xavier Initialization ==================
def U_xavier_limit(fan_in, fan_out):
    return np.sqrt(6.0 / (fan_in + fan_out))

# 2 -> 2 hidden
_lim_v = U_xavier_limit(2, 2)
U_V  = rng.uniform(-_lim_v, _lim_v, size=(2, 2))   # weights: input->hidden
U_bh = np.zeros(2, dtype=float)                    # biases: hidden

# 2 -> 1 output
_lim_w = U_xavier_limit(2, 1)
U_W  = rng.uniform(-_lim_w, _lim_w, size=(2, 1))   # weights: hidden->output
U_bo = np.zeros(1, dtype=float)                    # bias: output

# Momentum buffers (same shapes)
U_vW  = np.zeros_like(U_W);  U_vbo = np.zeros_like(U_bo)
U_vV  = np.zeros_like(U_V);  U_vbh = np.zeros_like(U_bh)

# ===================== Training (Batch) ====================
U_loss_curve = []
U_final_epoch = 0

for U_epoch in range(1, U_MAX_EPOCHS + 1):
    # ---- Forward (batch)
    U_Zh = U_X @ U_V + U_bh          # (4x2)
    U_Ah = U_sigmoid(U_Zh)           # (4x2)
    U_Zo = U_Ah @ U_W + U_bo         # (4x1)
    U_Y  = U_sigmoid(U_Zo)           # (4x1)

    # ---- Compute SSE for stopping
    U_E   = U_Y - U_T                # (4x1)
    U_SSE = float(np.sum(U_E**2))
    U_loss_curve.append(U_SSE)
    if U_SSE <= U_TARGET_SSE:
        U_final_epoch = U_epoch
        break

    # ---- Cross-Entropy gradients (with sigmoid): dL/dZo = (Y - T)
    U_dZo = (U_Y - U_T)                                  # (4x1)
    U_gW  = U_Ah.T @ U_dZo                               # (2x1)
    U_gbo = np.sum(U_dZo, axis=0)                        # (1,)
    U_dAh = U_dZo @ U_W.T                                # (4x2)
    U_dZh = U_dAh * U_dsigmoid(U_Ah)                     # (4x2)
    U_gV  = U_X.T @ U_dZh                                # (2x2)
    U_gbh = np.sum(U_dZh, axis=0)                        # (2,)

    # ---- Momentum updates: v = β v + α g ;  θ = θ - v
    U_vW  = U_MOMENTUM * U_vW  + U_ALPHA * U_gW
    U_vbo = U_MOMENTUM * U_vbo + U_ALPHA * U_gbo
    U_vV  = U_MOMENTUM * U_vV  + U_ALPHA * U_gV
    U_vbh = U_MOMENTUM * U_vbh + U_ALPHA * U_gbh

    U_W  -= U_vW
    U_bo -= U_vbo
    U_V  -= U_vV
    U_bh -= U_vbh

    U_final_epoch = U_epoch  # if we hit max epochs

# ======================= Evaluation ========================
def U_predict_row(x_row):
    zh = x_row @ U_V + U_bh
    ah = U_sigmoid(zh)
    zo = ah @ U_W + U_bo
    y  = U_sigmoid(zo)
    return float(y)

U_outputs = np.array([U_predict_row(U_X[i]) for i in range(len(U_X))])
U_bin_out = (U_outputs >= 0.5).astype(int)

print("\n=== A8: AND via Backprop (2–2–1, sigmoid, CE+Momentum) ===")
print(f"Final epoch: {U_final_epoch}")
print(f"Final SSE  : {U_SSE:.6f}")
print("Outputs (sigmoid):", np.round(U_outputs, 6).tolist())
print("Binarized (>=0.5):", U_bin_out.tolist())

print("\nFinal weights:")
print("V (input->hidden):")
print(np.round(U_V, 6))
print("b_h:", np.round(U_bh, 6))
print("W (hidden->output):")
print(np.round(U_W, 6))
print("b_o:", np.round(U_bo, 6))

# ===================== Plot & Save =========================
plt.figure(figsize=(6, 4))
plt.plot(U_loss_curve, linewidth=2)
plt.title("A8 – AND via Backprop: SSE vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Sum-Square Error")
plt.grid(True, linestyle="--", linewidth=0.6)
U_plot_path = os.path.join(U_OUTPUT_DIR, "U_A8_AND_loss.png")
plt.savefig(U_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plot:", U_plot_path)
