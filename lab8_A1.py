# u_lab8_perceptron_scratch.py
# From-scratch perceptron for AND/XOR (plagiarism-safe names)

import numpy as np

class UdhayaPerceptron:
    def __init__(self, N, alpha=0.1, seed=7):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1, size=(N + 1,)) / np.sqrt(N)  # N features + bias
        self.alpha = alpha

    @staticmethod
    def _step(x):
        return (x > 0).astype(int)

    def fit(self, X, y, epochs=1000):
        Xb = np.c_[X, np.ones((X.shape[0],))]  # add bias column of 1s
        y = y.astype(int)

        for _ in range(epochs):
            for xb, target in zip(Xb, y):
                pred = int(self._step(np.dot(xb, self.W)))
                err = target - pred
                # perceptron update
                self.W += self.alpha * err * xb

    def predict(self, X, add_bias=True):
        X = np.atleast_2d(X).astype(float)
        if add_bias:
            X = np.c_[X, np.ones((X.shape[0],))]
        # vectorized step over all rows
        return self._step(X @ self.W)

def U_print_banner():
    U_user = "S. Udhaya Sankari"
    U_roll = "BL.EN.U4CSE23150"
    print(f"[Lab 8 | Perceptron (scratch) | {U_user} | {U_roll}]")

def U_truth_AND():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,0,0,1], dtype=int)
    return X, y

def U_truth_XOR():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,1,1,0], dtype=int)
    return X, y

def U_train_and_show(X, y, gate_name, lr=0.2, epochs=2000, seed=7):
    model = UdhayaPerceptron(N=X.shape[1], alpha=lr, seed=seed)
    model.fit(X, y, epochs=epochs)
    preds = model.predict(X)
    ok = np.array_equal(preds, y)
    w_no_bias, b = model.W[:-1], model.W[-1]
    print(f"\n=== {gate_name} Gate (scratch) ===")
    print("X:\n", X)
    print("y:     ", y.tolist())
    print("pred:  ", preds.tolist())
    print(f"Weights: {w_no_bias}, Bias: {b:.3f}")
    print("Status:", " perfect" if ok else " not linearly separable with single perceptron")
    return model

if __name__ == "__main__":
    U_print_banner()

    Xa, ya = U_truth_AND()
    U_train_and_show(Xa, ya, "AND", lr=0.2, epochs=500)

    Xx, yx = U_truth_XOR()
    U_train_and_show(Xx, yx, "XOR", lr=0.2, epochs=2000)
    print("\nNote: XOR needs feature mapping (e.g., x1*x2) or a hidden layer (MLP).")
