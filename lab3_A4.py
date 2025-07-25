import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the labeled dataset
file_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
U_dataset = pd.read_csv(file_path)

# Filter for only classes 1 and 2
U_dataset = U_dataset[U_dataset['label'].isin([1, 2])]

# Split into features and labels
U_features = U_dataset.drop(columns=['label'])
U_labels = U_dataset['label']

# Perform train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    U_features, U_labels, test_size=0.3, random_state=42, stratify=U_labels
)

# Print the shapes
print("Shape of training features:", X_train.shape)
print("Shape of training labels:", y_train.shape)
print("Shape of testing features:", X_test.shape)
print("Shape of testing labels:", y_test.shape)
