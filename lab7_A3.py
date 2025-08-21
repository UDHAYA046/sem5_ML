# U_Lab07_A3.py
# Lab 07 – A3: Compare multiple classifiers and tabulate Train vs Test metrics
# Author: S. Udhaya Sankari
# Notes:
#   - Minimal, plagiarism-safe, with inline comments for viva.
#   - Exactly what A3 asks: fit listed classifiers, report Train/Test metrics in one table.
#   - No cross-validation; rare-class datasets are supported by safe splitting logic.

import os
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- classifiers required by A3 ---
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Optional gradient-boosting libraries (used only if installed)
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier  # type: ignore
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

# --------------------- user-configurable paths/labels ---------------------
U_DATA_PATH = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"   # input CSV
U_OUT_DIR   = r"C:\Users\Udhaya\sem5_ML\lab7_output_figures"         # output folder
U_TARGET    = "class"                                                # label column
U_SEED      = 42                                                     # reproducibility
U_TEST_SIZE = 0.20                                                   # 20% test split

# --------------------- helpers (no prints inside) ---------------------
def U_safe_split(df: pd.DataFrame, target_col: str, test_size: float, seed: int):
    """
    Create a train/test split. Use stratify iff every class has >= 2 samples.
    Otherwise, fall back to a non-stratified split (A3 doesn't require CV).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Check class counts; stratify only if all classes have >= 2 instances
    class_counts = y.value_counts()
    can_stratify = (class_counts.min() >= 2)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y if can_stratify else None
    )
    return Xtr, Xte, ytr, yte, can_stratify

def U_build_models(seed: int):
    """
    Build the dictionary of classifiers requested in A3.
    Use StandardScaler where appropriate via Pipeline.
    """
    models: Dict[str, Any] = {
        # SVM with RBF kernel + scaling
        "SVM_RBF": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=False, random_state=seed))
        ]),
        # Simple tree
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        # Random forest
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        # AdaBoost (with default base estimator)
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=seed),
        # Naïve Bayes (no scaling needed)
        "NaiveBayes": GaussianNB(),
        # MLP with modest capacity + scaling
        "MLP_128x64": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=seed))
        ]),
    }

    # Add XGBoost if available
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.1, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="mlogloss", tree_method="hist",
            random_state=seed, n_jobs=-1
        )

    # Add CatBoost if available (silent training)
    if _HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.1,
            verbose=False, random_state=seed
        )

    return models

def U_metric_row(y_true, y_pred, model_name: str, split_tag: str):
    """
    Compute Accuracy, Precision_macro, Recall_macro, F1_macro for one split (Train/Test).
    Returns a dict to be used in the final table.
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "Model": model_name,
        "Split": split_tag,
        "Accuracy": float(acc),
        "Precision_macro": float(prec),
        "Recall_macro": float(rec),
        "F1_macro": float(f1)
    }

# --------------------- main: all printing here only ---------------------
if __name__ == "__main__":
    os.makedirs(U_OUT_DIR, exist_ok=True)

    # 1) Load dataset
    df = pd.read_csv(U_DATA_PATH)

    # 2) Train/Test split (use stratify only if feasible)
    Xtr, Xte, ytr, yte, used_stratify = U_safe_split(df, U_TARGET, U_TEST_SIZE, U_SEED)

    # 3) Build required classifiers
    models = U_build_models(seed=U_SEED)

    # 4) Fit each model and collect Train/Test metrics
    rows: List[Dict[str, Any]] = []
    for name, est in models.items():
        est.fit(Xtr, ytr)

        # Evaluate on TRAIN
        yhat_tr = est.predict(Xtr)
        rows.append(U_metric_row(ytr, yhat_tr, name, "Train"))

        # Evaluate on TEST
        yhat_te = est.predict(Xte)
        rows.append(U_metric_row(yte, yhat_te, name, "Test"))

    # 5) Create a single table with Train vs Test rows per model
    results = pd.DataFrame(rows)

    # 6) Pivot to a wide, report-friendly table: columns per metric and split
    wide = results.pivot(index="Model", columns="Split", values=["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"])
    # Sorting models by Test F1 (descending) for easy reading
    if ("F1_macro", "Test") in wide.columns:
        wide = wide.sort_values(("F1_macro", "Test"), ascending=False)

    # 7) Save CSV (exact requirement of A3: tabulated Train vs Test metrics)
    out_csv = os.path.join(U_OUT_DIR, "Lab07_A3_results.csv")
    wide.to_csv(out_csv)

    # 8) Minimal console view
    print("\n=== Lab07 A3: Train vs Test Metrics (macro-averaged) ===")
    print(f"(Stratified split used: {used_stratify})")
    print(wide.round(4))
    print(f"\nSaved table -> {out_csv}")
