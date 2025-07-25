# lab3_O1_separate_plots.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Step 1: Generate normal distribution data
mean = 0
std_dev = 1
num_samples = 1000

data = np.random.normal(loc=mean, scale=std_dev, size=num_samples)

# Step 2: Plot histogram
plt.figure(figsize=(6, 4))
sns.histplot(data, bins=30, kde=False, color='skyblue', edgecolor='black')
plt.title("Histogram of Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 3: Plot normal distribution curve
plt.figure(figsize=(6, 4))
sns.kdeplot(data, color='red', label='KDE (Estimated PDF)')
x_vals = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
plt.plot(x_vals, norm.pdf(x_vals, mean, std_dev), 'b--', label='True Normal PDF')
plt.title("Normal Distribution Curve")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
