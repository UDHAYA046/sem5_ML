# lab3_A7.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the labeled dataset
file_path = "C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"
df = pd.read_csv(file_path)

# Split into features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Keep only classes 1 and 2 for binary classification
X = X[y <= 2]
y = y[y <= 2]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the kNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# ---------------------------------------------
# A7. Predicting behavior for test vectors
# ---------------------------------------------

# Predict labels for the entire test set
y_pred = neigh.predict(X_test)
print("\n=== Predictions for All Test Vectors ===")
print("Predicted Labels:", y_pred)
print("Actual Labels   :", y_test.values)

# Predict a single vector
sample_index = 0  # change this to try other test samples
test_vector = X_test.iloc[sample_index].values.reshape(1, -1)
predicted_class = neigh.predict(test_vector)

print("\n=== Prediction for a Single Test Vector ===")
print("Test Vector Index:", sample_index)
print("Predicted Class  :", predicted_class[0])
print("Actual Class     :", y_test.values[sample_index])
