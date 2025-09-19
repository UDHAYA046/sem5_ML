# ------------------------------------------------------------
# Lab 8 – A10 | Two-output MLP for AND & XOR (one-hot targets)
# Author : S. Udhaya Sankari | BL.EN.U4CSE23150
# Model  : 2–2–2 MLP
# Hidden : Sigmoid
# Output : Softmax (two nodes) with Cross-Entropy gradients
# Train  : Batch GD with Momentum (β=0.9), α=0.05
# Stop   : When SSE (sum over both outputs) <= 0.002 or epoch==1000
# Saves  : ...\lab8_output_figures\U_A10_AND_loss.png, U_A10_XOR_loss.png
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Config ----------------------------
U_ALPHA       = 0.05
U_MOMENTUM    = 0.90
U_MAX_EPOCHS  = 1000
U_TARGET_SSE  = 0.002
U_OUTPUT_DIR  = r"C:\Users\Udhaya\sem5_ML\lab8_output_figures"
os.makedirs(U_OUTPUT_DIR, exist_ok=True)

# ---------------------- Activations ------------------------
def U_sigmoid(z):       return 1.0 / (1.0 + np.exp(-z))
def U_dsigmoid(a):      return a * (1.0 - a)

def U_softmax(Z):
    """
    Stable softmax that works for both 2D (batch, classes)
    and 1D (classes,) inputs.
    """
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Zs = Z - np.max(Z)
        expZ = np.exp(Zs)
        return expZ / np.sum(expZ)
    else:
        Zs = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Zs)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

def U_xavier_limit(fan_in, fan_out):
    return np.sqrt(6.0 / (fan_in + fan_out))

# ---------------------- Data helpers -----------------------
U_X = np.array([[0., 0.],
                [0., 1.],
                [1., 0.],
                [1., 1.]], dtype=float)  # (4x2)

def one_hot_from_scalar(labels):
    # labels are 0/1; 0->[1,0], 1->[0,1]
    out = np.zeros((len(labels), 2), dtype=float)
    out[labels == 0, 0] = 1.0
    out[labels == 1, 1] = 1.0
    return out

# -------------------- Train function -----------------------
def train_gate(gate_name, seed=482):
    rng = np.random.default_rng(seed)

    if gate_name.upper() == "AND":
        y_scalar = np.array([0, 0, 0, 1])
    elif gate_name.upper() == "XOR":
        y_scalar = np.array([0, 1, 1, 0])
    else:
        raise ValueError("gate_name must be 'AND' or 'XOR'")

    U_T = one_hot_from_scalar(y_scalar)   # (4x2), one-hot

    # ----- Init: 2->2 hidden, 2->2 output (Xavier) + small random biases
    lim_v = U_xavier_limit(2, 2)
    U_V  = rng.uniform(-lim_v, lim_v, size=(2, 2))         # input->hidden
    U_bh = rng.uniform(-0.1, 0.1, size=2)                  # hidden bias

    lim_w = U_xavier_limit(2, 2)
    U_W  = rng.uniform(-lim_w, lim_w, size=(2, 2))         # hidden->output
    U_bo = rng.uniform(-0.1, 0.1, size=2)                  # output bias

    # Momentum buffers
    U_vW  = np.zeros_like(U_W);  U_vbo = np.zeros_like(U_bo)
    U_vV  = np.zeros_like(U_V);  U_vbh = np.zeros_like(U_bh)

    loss_curve, final_epoch, U_SSE = [], 0, None

    # ----- Batch training
    for epoch in range(1, U_MAX_EPOCHS + 1):
        # Forward
        Zh = U_X @ U_V + U_bh        # (4x2)
        Ah = U_sigmoid(Zh)           # (4x2)
        Zo = Ah @ U_W + U_bo         # (4x2)
        Y  = U_softmax(Zo)           # (4x2)  <-- softmax output

        # SSE for the lab's stop condition (sum over both outputs)
        E   = Y - U_T
        U_SSE = float(np.sum(E**2))
        loss_curve.append(U_SSE)
        if U_SSE <= U_TARGET_SSE:
            final_epoch = epoch
            break

        # CE with softmax: dL/dZo = (Y - T)
        dZo = (Y - U_T)                     # (4x2)
        gW  = Ah.T @ dZo                    # (2x2)
        gbo = np.sum(dZo, axis=0)           # (2,)
        dAh = dZo @ U_W.T                   # (4x2)
        dZh = dAh * U_dsigmoid(Ah)          # (4x2)
        gV  = U_X.T @ dZh                   # (2x2)
        gbh = np.sum(dZh, axis=0)           # (2,)

        # Momentum update: v = β v + α g;  θ = θ - v
        U_vW  = U_MOMENTUM * U_vW  + U_ALPHA * gW
        U_vbo = U_MOMENTUM * U_vbo + U_ALPHA * gbo
        U_vV  = U_MOMENTUM * U_vV  + U_ALPHA * gV
        U_vbh = U_MOMENTUM * U_vbh + U_ALPHA * gbh

        U_W  -= U_vW;   U_bo -= U_vbo
        U_V  -= U_vV;   U_bh -= U_vbh
        final_epoch = epoch

    # ----- Predictions & accuracy
    def predict_row(x):
        ah = U_sigmoid(x @ U_V + U_bh)               # (2,)
        zo = ah @ U_W + U_bo                          # (2,)
        y  = U_softmax(zo).ravel()                    # ensure 1D length-2
        return y

    Y_pred = np.vstack([predict_row(x) for x in U_X])       # (4x2)
    y_cls  = np.argmax(Y_pred, axis=1)                      # 0 or 1
    t_cls  = np.argmax(U_T, axis=1)
    acc    = float(np.mean(y_cls == t_cls))

    # ----- Print
    print(f"\n=== A10-{gate_name.upper()}: 2–2–2 MLP (sigmoid hidden, SOFTMAX output) ===")
    print(f"Final epoch: {final_epoch}")
    print(f"Final SSE  : {U_SSE:.6f}")
    print("Outputs (two nodes):")
    print(np.round(Y_pred, 6))
    print("Predicted class (argmax):", y_cls.tolist())
    print("Accuracy:", acc)

    print("\nFinal weights:")
    print("V (input->hidden):")
    print(np.round(U_V, 6))
    print("b_h:", np.round(U_bh, 6))
    print("W (hidden->output):")
    print(np.round(U_W, 6))
    print("b_o:", np.round(U_bo, 6))

    # ----- Plot & save
    plt.figure(figsize=(6, 4))
    plt.plot(loss_curve, linewidth=2)
    plt.title(f"A10 – {gate_name.upper()} (2-output): SSE vs Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Sum-Square Error")
    plt.grid(True, linestyle="--", linewidth=0.6)
    save_path = os.path.join(U_OUTPUT_DIR, f"U_A10_{gate_name.upper()}_loss.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", save_path)

    return {
        "epochs": final_epoch, "sse": U_SSE,
        "pred": Y_pred, "pred_cls": y_cls, "acc": acc,
        "V": U_V, "bh": U_bh, "W": U_W, "bo": U_bo,
        "plot": save_path
    }

# -------------------- Run both gates -----------------------
if __name__ == "__main__":
    res_and = train_gate("AND", seed=482)
    res_xor = train_gate("XOR", seed=947)   # different seed helps avoid symmetry
