# Import required libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load saved training and testing sets
X_train = pd.read_csv("X_train_lab3.csv")
y_train = pd.read_csv("y_train_lab3.csv").values.ravel()
X_test = pd.read_csv("X_test_lab3.csv")
y_test = pd.read_csv("y_test_lab3.csv").values.ravel()

# Drop 'filename' column if it exists (not a feature)
if 'filename' in X_train.columns:
    X_train = X_train.drop(columns=['filename'])
    X_test = X_test.drop(columns=['filename'])

# Initialize and train the classifier
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)

# Evaluate model using the test set
accuracy = model_knn.score(X_test, y_test)

# Display result
print("===================================")
print(f"kNN Classifier Accuracy: {accuracy * 100:.2f}%")
print("===================================")
