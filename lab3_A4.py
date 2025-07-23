import numpy as np
from sklearn.model_selection import train_test_split

# Sample dataset with two features and binary class labels
data = np.array([
    # Class 0 samples
    [1.0, 2.0, 0],
    [1.5, 1.8, 0],
    [1.2, 1.9, 0],
    # Class 1 samples
    [5.0, 8.0, 1],
    [6.0, 9.0, 1],
    [5.5, 8.5, 1]
])

def prepare_and_split_data(data, test_size=0.3, random_state=42):
    """
    Prepare data and split into train/test sets
    
    Args:
        data: Input dataset with features and labels
        test_size: Proportion of dataset to include in test split
        random_state: Seed for random number generator
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Extract features (X) and labels (y)
    X = data[:, :-1]  # All columns except last
    y = data[:, -1]   # Last column
    
    # Ensure binary classification by keeping only two classes
    unique_classes = np.unique(y)
    if len(unique_classes) > 2:
        print(f"Note: Dataset contains {len(unique_classes)} classes. Using first two classes.")
        class_mask = np.isin(y, unique_classes[:2])
        X = X[class_mask]
        y = y[class_mask]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserve class distribution
    )
    
    return X_train, X_test, y_train, y_test

# Main execution
if __name__ == "__main__":
    # Perform the train-test split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(data):.0%})")
    print(f"Test samples: {len(X_test)} ({len(X_test)/len(data):.0%})")
    
    # Print the splits
    print("\nTraining Features (X_train):")
    print(X_train)
    print("\nTraining Labels (y_train):")
    print(y_train)
    print("\nTest Features (X_test):")
    print(X_test)
    print("\nTest Labels (y_test):")
    print(y_test)
