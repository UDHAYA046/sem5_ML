# Import required libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load training and testing data
U_X_train = pd.read_csv("X_train_lab3.csv")
U_y_train = pd.read_csv("y_train_lab3.csv").values.ravel()  # Ensure it's 1D
U_X_test = pd.read_csv("X_test_lab3.csv")
U_y_test = pd.read_csv("y_test_lab3.csv").values.ravel()

# Initialize kNN classifier with k=3
U_knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the model
U_knn_model.fit(U_X_train, U_y_train)

# Predict on test set
U_y_pred = U_knn_model.predict(U_X_test)

# Display accuracy and classification report
print("\n===== kNN Classifier Results =====")
print(f"Accuracy on test set: {accuracy_score(U_y_test, U_y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(U_y_test, U_y_pred))
