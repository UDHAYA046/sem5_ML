# ============================================================
# Lab 10 – A2: PCA (retain 99% variance) + Model Comparison
# Author: S. Udhaya Sankari
# Rules: NO prints inside functions; prints/plots only in main.
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ---------------------- Data I/O & Cleaning (no prints) ----------------------

def U_load_csv(file_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return (X, y)."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_csv(p)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def U_numeric_only(
    X: pd.DataFrame,
    drop_if_present: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Return numeric-only feature frame.
    Optionally drop columns by name first (IDs like 'filename', 'file', 'path').
    Returns: (X_numeric, dropped_named, dropped_nonnumeric)
    """
    drop_if_present = drop_if_present or []
    drop_named = [c for c in drop_if_present if c in X.columns]
    X2 = X.drop(columns=drop_named, errors="ignore")
    X_num = X2.select_dtypes(include=[np.number]).copy()
    dropped_nonnumeric = [c for c in X2.columns if c not in X_num.columns]
    return X_num, drop_named, dropped_nonnumeric


# ---------------------- Robust split + scaling (no prints) ----------------------

def U_split_scale_safe(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, Dict[Any, int]]:
    """
    Robust splitter:
    - If all classes have >= 2 samples: stratified split.
    - If any class has < 2: keep those rare samples in TRAIN, split rest (stratified if possible).
    Returns scaled arrays, fitted scaler, and class counts.
    """
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    counts = y.value_counts().to_dict()
    rare_classes = [c for c, n in counts.items() if n < 2]

    if len(rare_classes) == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    else:
        rare_mask = y.isin(rare_classes)
        X_rare, y_rare = X[rare_mask], y[rare_mask]
        X_rest, y_rest = X[~rare_mask], y[~rare_mask]

        can_stratify_rest = (len(y_rest) > 0) and (y_rest.value_counts().min() >= 2)
        if can_stratify_rest:
            X_tr0, X_te0, y_tr0, y_te0 = train_test_split(
                X_rest, y_rest, test_size=test_size, stratify=y_rest, random_state=random_state
            )
        else:
            X_tr0, X_te0, y_tr0, y_te0 = train_test_split(
                X_rest, y_rest, test_size=test_size, shuffle=True, random_state=random_state
            )

        X_train = pd.concat([X_tr0, X_rare], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_tr0, y_rare], axis=0).reset_index(drop=True)
        X_test, y_test = X_te0.reset_index(drop=True), y_te0.reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled  = scaler.transform(X_test.values)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, counts


# ---------------------- PCA (no prints) ----------------------

def U_apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    variance_retained: float = 0.99,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA to X_train and transform both sets keeping given cumulative variance."""
    pca = PCA(n_components=variance_retained, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


# ---------------------- Model evaluation (no prints) ----------------------

def U_eval_classifier(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """Train & evaluate a classifier; return metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {"accuracy": acc, "f1_macro": f1m, "confusion_matrix": cm, "report": report, "y_pred": y_pred}


# ---------------------- Main (prints/plots only) ----------------------

if __name__ == "__main__":
    # ---- Config ----
    U_DATA_PATH   = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_TARGET_COL  = "class"     # change if different
    DROP_IF_PRESENT = ["filename", "file", "filepath", "path", "id"]  # common ID columns to drop first
    RNG = 42

    # 1) Load
    X_raw, y = U_load_csv(U_DATA_PATH, U_TARGET_COL)
    print(f"Loaded: {U_DATA_PATH}  |  X shape(before clean)={X_raw.shape}, y len={len(y)}")

    # 2) Keep numeric features only (drop IDs like 'filename')
    X, dropped_named, dropped_nonnum = U_numeric_only(X_raw, drop_if_present=DROP_IF_PRESENT)
    print(f"Dropped named columns (if present): {dropped_named}")
    print(f"Dropped non-numeric columns: {dropped_nonnum}")
    print(f"X shape(after numeric-only)={X.shape}")

    # 3) Robust split + scale
    X_tr, X_te, y_tr, y_te, scaler, class_counts = U_split_scale_safe(X, y, test_size=0.2, random_state=RNG)
    print("Class counts:", class_counts)
    print(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}")

    # 4) Baseline (no PCA)
    clf_log = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RNG)
    clf_rf  = RandomForestClassifier(n_estimators=200, random_state=RNG)
    base_log = U_eval_classifier(clf_log, X_tr, y_tr, X_te, y_te)
    base_rf  = U_eval_classifier(clf_rf,  X_tr, y_tr, X_te, y_te)

    print("\n=== Baseline (Without PCA) ===")
    print(f"Logistic Regression → Acc={base_log['accuracy']:.3f}, F1={base_log['f1_macro']:.3f}")
    print(f"Random Forest       → Acc={base_rf['accuracy']:.3f}, F1={base_rf['f1_macro']:.3f}")

    # 5) PCA @ 99% variance
    X_tr_pca, X_te_pca, pca = U_apply_pca(X_tr, X_te, variance_retained=0.99, random_state=RNG)
    print(f"\nPCA retained components: {X_tr_pca.shape[1]}  (cum. variance ≈ {pca.explained_variance_ratio_.sum():.3f})")

    # Plot cumulative variance curve
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance (retain 99%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6) Evaluate after PCA
    pca_log = U_eval_classifier(clf_log, X_tr_pca, y_tr, X_te_pca, y_te)
    pca_rf  = U_eval_classifier(clf_rf,  X_tr_pca, y_tr, X_te_pca, y_te)

    print("\n=== After PCA (Reduced Dimensionality) ===")
    print(f"Logistic Regression → Acc={pca_log['accuracy']:.3f}, F1={pca_log['f1_macro']:.3f}")
    print(f"Random Forest       → Acc={pca_rf['accuracy']:.3f}, F1={pca_rf['f1_macro']:.3f}")

    # 7) Side-by-side comparison table
    print("\n=== Performance Comparison ===")
    print(f"{'Model':<22} | {'Acc (Before)':<12} | {'Acc (After)':<12} | {'F1 (Before)':<12} | {'F1 (After)':<12}")
    print("-"*78)
    print(f"{'Logistic Regression':<22} | {base_log['accuracy']:<12.3f} | {pca_log['accuracy']:<12.3f} | {base_log['f1_macro']:<12.3f} | {pca_log['f1_macro']:<12.3f}")
    print(f"{'Random Forest':<22} | {base_rf['accuracy']:<12.3f} | {pca_rf['accuracy']:<12.3f} | {base_rf['f1_macro']:<12.3f} | {pca_rf['f1_macro']:<12.3f}")

    # 8) Save PCA cumulative variance plot beside CSV
    out_png = str(Path(U_DATA_PATH).with_suffix("")) + "_pca_cumvar.png"
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance (retain 99%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"\nSaved PCA cumulative variance plot → {out_png}")
