# ============================================================
# Lab 10 – A3: PCA (retain 95% variance) + Model Comparison
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ---------------------- Data I/O & Cleaning (no prints) ----------------------

def U_load_csv(file_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_csv(p)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def U_numeric_only(X: pd.DataFrame, drop_if_present: Optional[List[str]] = None
                   ) -> Tuple[pd.DataFrame, List[str], List[str]]:
    drop_if_present = drop_if_present or []
    drop_named = [c for c in drop_if_present if c in X.columns]
    X2 = X.drop(columns=drop_named, errors="ignore")
    X_num = X2.select_dtypes(include=[np.number]).copy()
    dropped_nonnum = [c for c in X2.columns if c not in X_num.columns]
    return X_num, drop_named, dropped_nonnum


# ---------------------- Robust split + scaling (no prints) ----------------------

def U_split_scale_safe(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, Dict[Any, int]]:
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


# ---------------------- PCA & evaluation utils (no prints) ----------------------

def U_apply_pca(X_train: np.ndarray, X_test: np.ndarray, variance_retained: float, random_state: int = 42
               ) -> Tuple[np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=variance_retained, random_state=random_state)
    return pca.fit_transform(X_train), pca.transform(X_test), pca

def U_eval_classifier(model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {"accuracy": acc, "f1_macro": f1m, "confusion_matrix": cm, "report": report, "y_pred": y_pred}

def U_cv_scores(model, X, y, n_splits=5, rng=42) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1m = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    return {"cv_acc_mean": acc.mean(), "cv_acc_std": acc.std(),
            "cv_f1m_mean": f1m.mean(), "cv_f1m_std": f1m.std()}


# ---------------------- Main (prints/plots only) ----------------------

if __name__ == "__main__":
    # ---- Config ----
    U_DATA_PATH    = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_TARGET_COL   = "class"
    DROP_IF_PRESENT = ["filename", "file", "filepath", "path", "id"]
    RNG = 42

    # 1) Load + keep numeric features only
    X_raw, y = U_load_csv(U_DATA_PATH, U_TARGET_COL)
    X, dropped_named, dropped_nonnum = U_numeric_only(X_raw, drop_if_present=DROP_IF_PRESENT)
    print(f"Loaded: {U_DATA_PATH} | X(before)={X_raw.shape} → X(numeric)={X.shape}, y={len(y)}")
    if dropped_named:    print("Dropped named columns:", dropped_named)
    if dropped_nonnum:   print("Dropped non-numeric columns:", dropped_nonnum)

    # 2) Robust split + scale
    X_tr, X_te, y_tr, y_te, scaler, class_counts = U_split_scale_safe(X, y, test_size=0.2, random_state=RNG)
    print("Class counts:", class_counts)
    print(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}")

    # 3) Baseline (no PCA)
    clf_log = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RNG)
    clf_rf  = RandomForestClassifier(n_estimators=200, random_state=RNG)
    base_log = U_eval_classifier(clf_log, X_tr, y_tr, X_te, y_te)
    base_rf  = U_eval_classifier(clf_rf,  X_tr, y_tr, X_te, y_te)
    print("\n=== Baseline (Without PCA) ===")
    print(f"Logistic Regression → Acc={base_log['accuracy']:.3f}, F1={base_log['f1_macro']:.3f}")
    print(f"Random Forest       → Acc={base_rf['accuracy']:.3f}, F1={base_rf['f1_macro']:.3f}")

    # 4) PCA @ 95% variance (A3 requirement)
    X_tr_pca95, X_te_pca95, pca95 = U_apply_pca(X_tr, X_te, variance_retained=0.95, random_state=RNG)
    print(f"\nPCA (95%) retained components: {X_tr_pca95.shape[1]}  (cum. variance ≈ {pca95.explained_variance_ratio_.sum():.3f})")

    # Plot and save cumulative variance curve (95%)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca95.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components"); plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance (retain 95%)")
    plt.grid(True); plt.tight_layout(); plt.show()

    out_png = str(Path(U_DATA_PATH).with_suffix("")) + "_pca95_cumvar.png"
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca95.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components"); plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance (retain 95%)")
    plt.grid(True); plt.tight_layout(); plt.savefig(out_png, dpi=300)
    print(f"Saved PCA(95%) cumulative variance plot → {out_png}")

    # 5) Evaluate models on PCA(95%) features
    pca95_log = U_eval_classifier(clf_log, X_tr_pca95, y_tr, X_te_pca95, y_te)
    pca95_rf  = U_eval_classifier(clf_rf,  X_tr_pca95, y_tr, X_te_pca95, y_te)

    print("\n=== After PCA (95% variance) ===")
    print(f"Logistic Regression → Acc={pca95_log['accuracy']:.3f}, F1={pca95_log['f1_macro']:.3f}")
    print(f"Random Forest       → Acc={pca95_rf['accuracy']:.3f}, F1={pca95_rf['f1_macro']:.3f}")

    # 6) Optional: 5-fold CV on TRAIN (stability for small n)
    cv_log_95 = U_cv_scores(clf_log, X_tr_pca95, y_tr, n_splits=5, rng=RNG)
    cv_rf_95  = U_cv_scores(clf_rf,  X_tr_pca95, y_tr, n_splits=5, rng=RNG)
    print("\n=== 5-fold CV on Train (+PCA 95%) mean ± std ===")
    print(f"LogReg : Acc={cv_log_95['cv_acc_mean']:.3f}±{cv_log_95['cv_acc_std']:.3f} | "
          f"F1m={cv_log_95['cv_f1m_mean']:.3f}±{cv_log_95['cv_f1m_std']:.3f}")
    print(f"RF     : Acc={cv_rf_95['cv_acc_mean']:.3f}±{cv_rf_95['cv_acc_std']:.3f} | "
          f"F1m={cv_rf_95['cv_f1m_mean']:.3f}±{cv_rf_95['cv_f1m_std']:.3f}")

    # 7) Side-by-side comparison table (Baseline vs PCA95)
    print("\n=== Performance Comparison (Baseline vs PCA 95%) ===")
    print(f"{'Model':<22} | {'Acc (Before)':<12} | {'Acc (PCA95)':<12} | {'F1 (Before)':<12} | {'F1 (PCA95)':<12}")
    print("-"*78)
    print(f"{'Logistic Regression':<22} | {base_log['accuracy']:<12.3f} | {pca95_log['accuracy']:<12.3f} | "
          f"{base_log['f1_macro']:<12.3f} | {pca95_log['f1_macro']:<12.3f}")
    print(f"{'Random Forest':<22} | {base_rf['accuracy']:<12.3f} | {pca95_rf['accuracy']:<12.3f} | "
          f"{base_rf['f1_macro']:<12.3f} | {pca95_rf['f1_macro']:<12.3f}")
