# ---------------------------- MODULE IMPORTS ----------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---------------------------- FUNCTION DEFINITIONS ----------------------------

def U_train_knn_model(U_X_train, U_y_train, U_k_value=3):
    """
    Trains a k-Nearest Neighbors classifier.
    """
    U_knn = KNeighborsClassifier(n_neighbors=U_k_value)
    U_knn.fit(U_X_train, U_y_train.values.ravel())
    return U_knn

def U_compute_metrics(U_model, U_X_data, U_y_actual):
    """
    Computes confusion matrix, precision, recall, and F1-score.
    """
    U_y_predicted = U_model.predict(U_X_data)
    U_conf_matrix = confusion_matrix(U_y_actual, U_y_predicted)
    U_precision = precision_score(U_y_actual, U_y_predicted, average='weighted', zero_division=0)
    U_recall = recall_score(U_y_actual, U_y_predicted, average='weighted', zero_division=0)
    U_f1 = f1_score(U_y_actual, U_y_predicted, average='weighted', zero_division=0)

    return {
        "Confusion Matrix": U_conf_matrix,
        "Precision": U_precision,
        "Recall": U_recall,
        "F1-Score": U_f1
    }

def U_plot_confusion_matrix(U_matrix, U_title, U_labels, U_filename):
    """
    Plots and saves a confusion matrix as a high-resolution PNG.
    """
    U_output_dir = r"C:\Users\Udhaya\sem5_ML\lab4_output_figures"
    os.makedirs(U_output_dir, exist_ok=True)  # Create folder if missing

    plt.figure(figsize=(6, 5))
    sns.heatmap(U_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=U_labels, yticklabels=U_labels,
                cbar=False, linewidths=0.5, linecolor='black')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(U_title)
    plt.tight_layout()

    U_save_path = os.path.join(U_output_dir, U_filename)
    plt.savefig(U_save_path, dpi=400)  # Save at high resolution
    plt.show()
    plt.close()

# ---------------------------- MAIN PROGRAM ----------------------------

if __name__ == "__main__":
    # Load training and test data
    U_X_train = pd.read_csv("train_features.csv")
    U_y_train = pd.read_csv("train_labels.csv")
    U_X_test = pd.read_csv("test_features.csv")
    U_y_test = pd.read_csv("test_labels.csv")

    # Train model
    U_classifier = U_train_knn_model(U_X_train, U_y_train, U_k_value=3)

    # Evaluate training data
    U_train_results = U_compute_metrics(U_classifier, U_X_train, U_y_train)

    # Evaluate test data
    U_test_results = U_compute_metrics(U_classifier, U_X_test, U_y_test)

    # Print metrics
    print(" Training Set Evaluation:")
    print("Confusion Matrix:\n", U_train_results["Confusion Matrix"])
    print("Precision:", U_train_results["Precision"])
    print("Recall:", U_train_results["Recall"])
    print("F1-Score:", U_train_results["F1-Score"])

    print("\n Test Set Evaluation:")
    print("Confusion Matrix:\n", U_test_results["Confusion Matrix"])
    print("Precision:", U_test_results["Precision"])
    print("Recall:", U_test_results["Recall"])
    print("F1-Score:", U_test_results["F1-Score"])

    # Plot and save confusion matrices
    U_class_names = ["Class 1", "Class 2"]
    U_plot_confusion_matrix(U_train_results["Confusion Matrix"],
                            "Training Set Confusion Matrix",
                            U_class_names,
                            "conf_matrix_train.png")

    U_plot_confusion_matrix(U_test_results["Confusion Matrix"],
                            "Test Set Confusion Matrix",
                            U_class_names,
                            "conf_matrix_test.png")
