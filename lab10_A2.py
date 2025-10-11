# ============================================================
# Lab 10 – A2: PCA for Dimensionality Reduction (retain 99% variance)
# Author: S. Udhaya Sankari
# Rules followed: all logic in functions; prints/plots only in main.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from typing import Tuple, Dict, Any

# ---------------------- Functions (No prints) ----------------------

def U_load_dataset(file_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split into features and target."""
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def U_split_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Train-test split and feature scaling."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def U_apply_pca(X_train: np.ndarray, X_test: np.ndarray, variance_retained: float = 0.99) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Apply PCA keeping cumulative variance at given threshold."""
    pca = PCA(n_components=variance_retained, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def U_eval_classifier(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """Train classifier and return metrics (no prints)."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm, "report": report, "y_pred": y_pred}

# ---------------------- Main (Prints & Plots) ----------------------

if __name__ == "__main__":
    # Step 1: File details
    U_DATA_PATH = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_TARGET_COL = "class"   # change if your label column differs

    # Step 2: Load and scale
    X, y = U_load_dataset(U_DATA_PATH, U_TARGET_COL)
    X_train, X_test, y_train, y_test, scaler = U_split_scale(X, y)

    print(f"Original feature shape: {X_train.shape[1]} features")

    # Step 3: Baseline (before PCA)
    clf1 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    clf2 = RandomForestClassifier(n_estimators=200, random_state=42)
    base_log = U_eval_classifier(clf1, X_train, y_train, X_test, y_test)
    base_rf = U_eval_classifier(clf2, X_train, y_train, X_test, y_test)

    print("\n=== Baseline (Without PCA) ===")
    print(f"Logistic Regression → Acc={base_log['accuracy']:.3f}, F1={base_log['f1_macro']:.3f}")
    print(f"Random Forest        → Acc={base_rf['accuracy']:.3f}, F1={base_rf['f1_macro']:.3f}")

    # Step 4: Apply PCA (retain 99% variance)
    X_train_pca, X_test_pca, pca_model = U_apply_pca(X_train, X_test, variance_retained=0.99)

    print(f"\nAfter PCA: {X_train_pca.shape[1]} components retained (≈99% variance)")

    # Step 5: Plot cumulative variance curve
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca_model.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Step 6: Evaluate models on PCA-transformed features
    pca_log = U_eval_classifier(clf1, X_train_pca, y_train, X_test_pca, y_test)
    pca_rf = U_eval_classifier(clf2, X_train_pca, y_train, X_test_pca, y_test)

    print("\n=== After PCA (Reduced Dimensionality) ===")
    print(f"Logistic Regression → Acc={pca_log['accuracy']:.3f}, F1={pca_log['f1_macro']:.3f}")
    print(f"Random Forest        → Acc={pca_rf['accuracy']:.3f}, F1={pca_rf['f1_macro']:.3f}")

    # Step 7: Compare results
    print("\n=== Performance Comparison ===")
    print(f"{'Model':<22} | {'Acc (Before)':<12} | {'Acc (After)':<12} | {'F1 (Before)':<12} | {'F1 (After)':<12}")
    print("-"*70)
    print(f"{'Logistic Regression':<22} | {base_log['accuracy']:<12.3f} | {pca_log['accuracy']:<12.3f} | {base_log['f1_macro']:<12.3f} | {pca_log['f1_macro']:<12.3f}")
    print(f"{'Random Forest':<22} | {base_rf['accuracy']:<12.3f} | {pca_rf['accuracy']:<12.3f} | {base_rf['f1_macro']:<12.3f} | {pca_rf['f1_macro']:<12.3f}")
