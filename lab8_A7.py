import numpy as np

# === Data (XOR) ===
U_X = np.array([[0,0],[0,1],[1,0],[1,1]])
U_T = np.array([[0],[1],[1],[0]])

# Add bias term
U_X_bias = np.hstack([U_X, np.ones((U_X.shape[0],1))])

# === Pseudo-inverse solution ===
U_W_pinv = np.linalg.pinv(U_X_bias) @ U_T
print("Weights from pseudo-inverse:", U_W_pinv.ravel())

# Predictions
U_Y_pinv = U_X_bias @ U_W_pinv
print("Outputs:", U_Y_pinv.ravel())

# Thresholding at 0.5
U_Y_bin = (U_Y_pinv >= 0.5).astype(int)
print("Binarized predictions:", U_Y_bin.ravel())

# Compare with perceptron results (A5)
print("\nComparison:")
print("Perceptron (A5): SSE ~ 2.0 (no convergence)")
print("Pseudo-inverse (A7): SSE =", np.sum((U_T - U_Y_pinv)**2))
