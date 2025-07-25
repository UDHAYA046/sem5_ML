import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load dataset
file_path = "C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"
df = pd.read_csv(file_path)

# Drop unnecessary column
df = df.drop(columns=['filename'])  # Adjust if needed

# Features and target
X = df.drop("class", axis=1)
y = df["class"]

# Filter for binary classification (classes 1 and 2 only)
X = X[y <= 2]
y = y[y <= 2]

# Binarize labels (class 1 = 0, class 2 = 1)
y_bin = label_binarize(y, classes=[1, 2]).ravel()

# Split data
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42)

# Train kNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train_bin)

# Predict probabilities
y_probs = knn.predict_proba(X_test)[:, 1]  # Probability of class 1

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_probs)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'kNN (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - kNN Classifier (Binary: Class 1 vs 2)')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()
