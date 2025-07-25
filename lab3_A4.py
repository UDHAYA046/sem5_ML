import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
file_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
U_dataset = pd.read_csv(file_path)

# Filter only two classes (1 and 2)
U_dataset = U_dataset[U_dataset['class'].isin([1, 2])]

# Separate features and labels
U_features = U_dataset.drop(['filename', 'class'], axis=1)
U_labels = U_dataset['class']

# Split dataset (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(U_features, U_labels, test_size=0.3, random_state=42)

# Save output to CSV for verification (optional)
X_train.to_csv("train_features.csv", index=False)
X_test.to_csv("test_features.csv", index=False)
y_train.to_csv("train_labels.csv", index=False)
y_test.to_csv("test_labels.csv", index=False)

print("Dataset split completed successfully.")
print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size : {X_test.shape[0]} samples")
