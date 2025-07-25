# lab3_A9.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"
df = pd.read_csv(file_path)

# Drop non-numeric column
df = df.drop(columns=['filename'])

# Separate features and labels
X = df.drop("class", axis=1)
y = df["class"]

# Filter only class 1 and 2
X = X[y <= 2]
y = y[y <= 2]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train kNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Confusion Matrices
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix
def plot_cm(cm, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

print("\n=== Training Classification Report ===")
print(classification_report(y_train, y_train_pred, digits=4))

print("\n=== Testing Classification Report ===")
print(classification_report(y_test, y_test_pred, digits=4))

# Plot confusion matrices
plot_cm(train_cm, "Training Data")
plot_cm(test_cm, "Testing Data")

# Summary accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")

# Inference
if train_acc - test_acc > 0.15:
    print("\n>>> Model is likely OVERFITTING")
elif test_acc - train_acc > 0.15:
    print("\n>>> Model is likely UNDERFITTING")
else:
    print("\n>>> Model is likely REGULAR-FIT (Generalizing well)")
