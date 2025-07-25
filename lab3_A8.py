# lab3_A8.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
file_path = "C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"
df = pd.read_csv(file_path)

# Drop non-numeric column
df = df.drop(columns=['filename'])  # Make sure 'filename' column exists

# Separate features and labels
X = df.drop("class", axis=1)
y = df["class"]

# Filter only class 1 and 2
X = X[y <= 2]
y = y[y <= 2]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Store accuracy values
k_values = list(range(1, 12))
accuracies = []

print("\n=== Accuracy Report ===")
for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k = {k:2d} -> Accuracy = {acc:.4f}")

# Plot accuracy vs k
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title("Accuracy of kNN Classifier for different k values")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()
