# U_Lab07_A2.py
# Lab 07 – A2: Hyperparameter tuning with RandomizedSearchCV
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
from scipy.stats import loguniform, randint

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ----------------- config -----------------
U_DATA_PATH = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
U_OUT_DIR   = r"C:\Users\Udhaya\sem5_ML\lab7_output_figures"
U_TARGET    = "class"
U_SEED      = 42

# ----------------- helpers -----------------
def U_load_split(csv_path: str, target_col: str, seed: int = 42):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

def U_build_search_space(seed: int = 42) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    svm = Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(probability=True, random_state=seed))
    ])
    svm_space = {
        "clf__C": loguniform(1e-3, 1e3),
        "clf__gamma": loguniform(1e-4, 1e0),
        "clf__kernel": ["rbf"]
    }
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    rf_space = {
        "n_estimators": randint(150, 600),
        "max_depth": randint(3, 30),
        "max_features": ["sqrt", "log2", None]
    }
    return {"SVM_RBF": (svm, svm_space), "RandomForest": (rf, rf_space)}

def U_run_randomized_search(Xtr, ytr, Xte, yte, search_specs, seed=42, n_iter=25, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: List[Dict[str, Any]] = []
    for name, (est, space) in search_specs.items():
        rs = RandomizedSearchCV(
            estimator=est,
            param_distributions=space,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            random_state=seed,
            n_jobs=-1,
            refit=True
        )
        rs.fit(Xtr, ytr)
        best = rs.best_estimator_
        yhat = best.predict(Xte)
        acc = accuracy_score(yte, yhat)
        f1m = f1_score(yte, yhat, average="macro")
        prec, rec, _, _ = precision_recall_fscore_support(yte, yhat, average="macro", zero_division=0)
        rows.append({
            "Model": name,
            "BestParams": rs.best_params_,
            "CV5_F1_macro": float(rs.best_score_),
            "Test_Accuracy": float(acc),
            "Test_Precision_macro": float(prec),
            "Test_Recall_macro": float(rec),
            "Test_F1_macro": float(f1m)
        })
    return pd.DataFrame(rows).sort_values("Test_F1_macro", ascending=False)

# ----------------- main block -----------------
if __name__ == "__main__":
    os.makedirs(U_OUT_DIR, exist_ok=True)

    # 1. Data split
    Xtr, Xte, ytr, yte = U_load_split(U_DATA_PATH, U_TARGET, seed=U_SEED)

    # 2. Define models + spaces
    search_specs = U_build_search_space(seed=U_SEED)

    # 3. Run RandomizedSearchCV
    results = U_run_randomized_search(Xtr, ytr, Xte, yte, search_specs, seed=U_SEED)

    # 4. Save results table
    save_path = os.path.join(U_OUT_DIR, "Lab07_A2_results.csv")
    results.to_csv(save_path, index=False)

    # 5.  console summary
    print("\n=== Lab07 A2: RandomizedSearchCV results ===")
    print(results[["Model", "CV5_F1_macro", "Test_Accuracy", "Test_F1_macro"]])
    print(f"\nSaved CSV -> {save_path}")
