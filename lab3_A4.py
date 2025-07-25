# A4. Split Dataset into Training and Testing Sets
# Udhaya Sankari | Roll No: BL.EN.U4CSE23150

import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
ud_file_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
ud_df = pd.read_csv(ud_file_path)

# Step 2: Separate features (X) and labels (y)
X = ud_df.drop(columns=['filename', 'class'], errors='ignore')
y = ud_df['class']

# Step 3: Split into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Show result shapes
print(f" Total samples: {len(X)}")
print(f" Training samples: {len(X_train)}")
print(f" Testing samples: {len(X_test)}")

# Optional: preview the split data
print("\n--- First 5 Training Labels ---\n", y_train.head())
print("\n--- First 5 Test Labels ---\n", y_test.head())
