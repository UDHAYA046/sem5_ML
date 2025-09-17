# Lab 8 – A6: Perceptron (sigmoid) for High/Low transaction classification
# Author: S. Udhaya Sankari | Roll: BL.EN.U4CSE23150

import numpy as np
import os
import matplotlib.pyplot as plt

# ------------ output folder ------------
U_OUTDIR = r"C:\Users\Udhaya\sem5_ML\lab8_output_figures"
os.makedirs(U_OUTDIR, exist_ok=True)

# ------------ dataset (from prompt) ------------
# Features: [Candies, Mangoes(kg), MilkPackets, Payment(Rs)]
U_X_raw = np.array([
    [20, 6, 2, 386],  # C_1
    [16, 3, 6, 289],  # C_2
    [27, 6, 2, 393],  # C_3
    [19, 1, 2, 110],  # C_4
    [24, 4, 2, 280],  # C_5
    [22, 1, 5, 167],  # C_6
    [15, 4, 2, 271],  # C_7
    [18, 4, 2, 274],  # C_8
    [21, 1, 4, 148],  # C_9
    [16, 2, 4, 198],  # C_10
], dtype=float)

# Labels: "Yes"->1, "No"->0
U_y = np.array([1,1,1,0,1,0,1,1,0,0], dtype=float)

U_customers = np.array(["C_1","C_2","C_3","C_4","C_5","C_6","C_7","C_8","C_9","C_10"])

# ------------ preprocessing ------------
def U_minmax_scale(X):
    mn = X.min(axis=0); mx = X.max(axis=0)
    # avoid divide-by-zero
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    return (X - mn) / denom, mn, mx

U_X, U_mn, U_mx = U_minmax_scale(U_X_raw)

# ------------ model: logistic perceptron ------------
def U_sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z))

def U_train_logistic(X, y, lr=0.1, epochs=5000, seed=7, l2=0.0):
    """Binary logistic regression trained by gradient descent.
       Returns weights (including bias as last term) and loss history."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(X.shape[1]+1,))  # [w1..wd, b]
    hist = []
    for ep in range(1, epochs+1):
        # add bias feature of 1s
        Xb = np.c_[X, np.ones((X.shape[0],))]
        z = Xb @ W
        p = U_sigmoid(z)

        # Binary cross-entropy loss with optional L2
        eps = 1e-12
        loss = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)) + 0.5*l2*np.sum(W[:-1]**2)
        hist.append(loss)

        # Gradient
        grad = Xb.T @ (p - y) / X.shape[0]
        grad[:-1] += l2 * W[:-1]  # L2 only on weights, not bias
        W -= lr * grad
    return W, np.array(hist)

def U_predict(X, W, thr=0.5):
    Xb = np.c_[X, np.ones((X.shape[0],))]
    probs = U_sigmoid(Xb @ W)
    return (probs >= thr).astype(int), probs

# ------------ train ------------
U_lr = 0.1      # you can change, as allowed by A6
U_epochs = 5000

U_W, U_loss = U_train_logistic(U_X, U_y, lr=U_lr, epochs=U_epochs, seed=7, l2=0.0)
U_pred, U_prob = U_predict(U_X, U_W, thr=0.5)
U_acc = (U_pred == U_y).mean()

print("[Lab 8 – A6 | Sigmoid Perceptron]")
print(f"Learning rate = {U_lr}, Epochs = {U_epochs}")
print(f"Final weights (w1..w4, b): {np.round(U_W, 4).tolist()}")
print(f"Training accuracy: {U_acc*100:.1f}%")

print("\nCustomer-wise predictions:")
for cid, ytrue, phat in zip(U_customers, U_y.astype(int), U_prob):
    print(f"{cid:>4s}  target={ytrue}  prob_high={phat:.3f}  pred={'Yes' if phat>=0.5 else 'No'}")

# ------------ loss plot ------------
plt.figure()
plt.plot(np.arange(1, U_loss.size+1), U_loss)
plt.xlabel("Epochs"); plt.ylabel("Binary Cross-Entropy")
plt.title("A6: Training loss (sigmoid perceptron)")
plt.grid(True); plt.tight_layout()
save_path = os.path.join(U_OUTDIR, "U_A6_sigmoid_loss.png")
plt.savefig(save_path, dpi=150)
print(f"\nSaved plot: {save_path}")
