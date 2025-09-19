# ------------------------------------------------------------
# Lab 8 – A11 | AND & XOR using scikit-learn MLPClassifier
# Author : S. Udhaya Sankari | BL.EN.U4CSE23150
# Model  : 2 -> 2 (hidden) -> 1
# Hidden : logistic (sigmoid)
# Solver : SGD (lr=0.05, momentum=0.9), max_iter=1000
# Output : binary (class 0/1); we also compute SSE on prob outputs
# Saves  : ...\lab8_output_figures\U_A11_AND_loss.png
#          ...\lab8_output_figures\U_A11_XOR_loss.png
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- Config ----------
U_OUTPUT_DIR = r"C:\Users\Udhaya\sem5_ML\lab8_output_figures"
os.makedirs(U_OUTPUT_DIR, exist_ok=True)

# ---------- Data ----------
U_X = np.array([[0., 0.],
                [0., 1.],
                [1., 0.],
                [1., 1.]], dtype=float)

def train_gate(gate_name, y_scalar, seed):
    print(f"\n=== A11-{gate_name}: scikit-learn MLPClassifier (2-2-1, sigmoid, SGD) ===")

    clf = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation="logistic",          # sigmoid
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=0.05,        # α = 0.05 (as in A1/A8/A9)
        momentum=0.9,
        max_iter=1000,
        shuffle=True,
        random_state=seed,
        n_iter_no_change=1000,          # don’t early-stop before 1000
        verbose=False
    )

    clf.fit(U_X, y_scalar)

    # Predictions & probabilities
    y_pred = clf.predict(U_X)
    y_prob = clf.predict_proba(U_X)[:, 1]  # P(class=1)

    # Metrics
    acc = accuracy_score(y_scalar, y_pred)
    sse = float(np.sum((y_scalar - y_prob) ** 2))

    print("Targets:      ", y_scalar.tolist())
    print("Prob(class=1):", np.round(y_prob, 6).tolist())
    print("Predictions:  ", y_pred.tolist())
    print("Accuracy:     ", acc)
    print("Final SSE(on probs):", round(sse, 6))

    # Weights
    # coefs_[0]: (2x2) input->hidden,  coefs_[1]: (2x1) hidden->out
    print("\nWeights (input->hidden):")
    print(np.round(clf.coefs_[0], 6))
    print("Bias hidden:", np.round(clf.intercepts_[0], 6))
    print("Weights (hidden->out):")
    print(np.round(clf.coefs_[1], 6))
    print("Bias out:", np.round(clf.intercepts_[1], 6))

    # Loss curve plot (cross-entropy)
    plt.figure(figsize=(6, 4))
    plt.plot(clf.loss_curve_, linewidth=2)
    plt.title(f"A11 – {gate_name}: scikit-learn MLP loss")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", linewidth=0.6)
    out_path = os.path.join(U_OUTPUT_DIR, f"U_A11_{gate_name}_loss.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved loss plot:", out_path)

    # Confusion matrix & report (nice for the report appendix)
    print("\nConfusion Matrix:\n", confusion_matrix(y_scalar, y_pred))
    print("\nClassification Report:\n", classification_report(y_scalar, y_pred, digits=4))

    return {
        "acc": acc,
        "sse": sse,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "coefs": clf.coefs_,
        "intercepts": clf.intercepts_,
        "loss_plot": out_path,
        "loss_curve": clf.loss_curve_,
    }

if __name__ == "__main__":
    # AND and XOR labels
    y_AND = np.array([0, 0, 0, 1], dtype=int)
    y_XOR = np.array([0, 1, 1, 0], dtype=int)

    res_and = train_gate("AND", y_AND, seed=482)
    res_xor = train_gate("XOR", y_XOR, seed=947)
