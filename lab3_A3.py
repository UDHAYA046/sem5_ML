import numpy as np
import matplotlib.pyplot as plt

# Sample dataset with two features and class labels
data = np.array([
    [1.0, 2.0, 0],
    [1.5, 1.8, 0],
    [1.2, 1.9, 0],
    [5.0, 8.0, 1],
    [6.0, 9.0, 1],
    [5.5, 8.5, 1]
])

def calculate_minkowski(x, y, r):
    """Calculate Minkowski distance between vectors x and y for order r"""
    return np.sum(np.abs(x - y) ** r) ** (1 / r)

# Select first two samples as our feature vectors
vec1 = data[0, :2]  # First sample's features [1.0, 2.0]
vec2 = data[1, :2]  # Second sample's features [1.5, 1.8]

# Calculate distances for r=1 to 10
r_values = range(1, 11)
distances = [calculate_minkowski(vec1, vec2, r) for r in r_values]

# Set up the plot
plt.figure(figsize=(10, 6))
plt.plot(r_values, distances, 'bo-', linewidth=2, markersize=8)
plt.title('Minkowski Distance Analysis (r=1 to 10)', fontsize=14)
plt.xlabel('r value (order)', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.xticks(r_values)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Add distance values as annotations
for r, dist in zip(r_values, distances):
    plt.annotate(f'{dist:.2f}', (r, dist), textcoords="offset points", 
                 xytext=(0,10), ha='center')

plt.show()

# Print results in a formatted table
print("\nMinkowski Distance Analysis:")
print("----------------------------")
print(f"Vector 1: {vec1}")
print(f"Vector 2: {vec2}\n")
print("r\tDistance")
print("----------------")
for r, dist in zip(r_values, distances):
    print(f"{r}\t{dist:.4f}")
