import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
file_path = "C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"
df = pd.read_csv(file_path)

# Drop non-numeric column
df = df.drop(columns=["filename"])  # Adjust if column name differs

# Separate features and labels
X = df.drop("class", axis=1)
y = df["class"]

# Filter only classes 1 and 2
X = X[y <= 2]
y = y[y <= 2]

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define distance metrics to test
distance_metrics = [
    ("euclidean", {}),
    ("manhattan", {}),
    ("chebyshev", {}),
    ("minkowski", {"p": 3}),
    ("minkowski", {"p": 4})
]

accuracies = []

# Evaluate each distance metric
for metric_name, params in distance_metrics:
    model = KNeighborsClassifier(n_neighbors=3, metric=metric_name, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append((f"{metric_name} {params}", acc))
    print(f"Metric: {metric_name} {params} -> Accuracy: {acc:.4f}")

# Plot the results
labels, acc_vals = zip(*accuracies)
plt.figure(figsize=(8, 5))
plt.bar(labels, acc_vals, color='teal')
plt.title("Accuracy of kNN with Different Distance Metrics")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
