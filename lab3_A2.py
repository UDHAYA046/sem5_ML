import numpy as np
import matplotlib.pyplot as plt

# Sample dataset: Each row represents a sample, and the last column represents the class label
data = np.array([
    [1.0, 2.0, 0],
    [1.5, 1.8, 0],
    [1.2, 1.9, 0],
    [5.0, 8.0, 1],
    [6.0, 9.0, 1],
    [5.5, 8.5, 1]
])

# Select a feature to analyze (e.g., the first feature)
feature = data[:, 0]  # Taking the first feature from the dataset

# Calculate the mean and variance of the selected feature
mean_feature = np.mean(feature)
variance_feature = np.var(feature)

# Generate histogram data
hist, bins = np.histogram(feature, bins=5)  # 5 buckets for the histogram

# Plotting the histogram
plt.figure(figsize=(8, 5))
plt.hist(feature, bins=5, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Selected Feature')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.axvline(mean_feature, color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# Print the results
print("Mean of the selected feature:", mean_feature)
print("Variance of the selected feature:", variance_feature)
