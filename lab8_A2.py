# u_lab8_A2_custom_perceptron.py
import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def U_step(x):
    return 1 if x >= 0 else 0

# Training data (AND gate)
U_X = np.array([[0,0],[0,1],[1,0],[1,1]])
U_y = np.array([0,0,0,1])

# Initialize weights [w1, w2, bias]
U_W = np.array([0.2, -0.75, 10.0])   # [w1, w2, bias]
U_alpha = 0.05
U_errors = []

# Training loop
for epoch in range(1000):
    total_error = 0
    for xi, target in zip(U_X, U_y):
        x_with_bias = np.append(xi, 1)    # add bias input
        y_hat = U_step(np.dot(x_with_bias, U_W))
        error = target - y_hat
        U_W += U_alpha * error * x_with_bias
        total_error += (error ** 2) / 2.0  # sum-square error

    U_errors.append(total_error)
    if total_error <= 0.002:
        print(f"Converged at epoch {epoch+1}")
        break

print("Final Weights:", U_W)

# Plot error vs epochs
plt.plot(range(1, len(U_errors)+1), U_errors, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Sum-Square Error")
plt.title("Error Convergence for AND Gate (Custom Perceptron)")
plt.grid(True)
plt.show()
