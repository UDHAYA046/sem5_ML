# U_Lab07_A2.py  (patched for rare-class handling)
# Lab 07 – A2: RandomizedSearchCV with robust stratified split

import os
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
U_TEST_SIZE = 0.2

# ----------------- helpers -----------------
def U_build_search_space(seed: int = 42) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    svm = Pipeline([("scale", StandardScaler()), ("clf", SVC(probability=True, random_state=seed))])
    svm_space = {"clf__C": loguniform(1e-3, 1e3), "clf__gamma": loguniform(1e-4, 1e0), "clf__kernel": ["rbf"]}
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    rf_space = {"n_estimators": randint(150, 600), "max_depth": randint(3, 30), "max_features": ["sqrt", "log2", None]}
    return {"SVM_RBF": (svm, svm_space), "RandomForest": (rf, rf_space)}

def U_run_randomized_search(Xtr, ytr, Xte, yte, search_specs, seed=42, n_iter=25, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: List[Dict[str, Any]] = []
    for name, (est, space) in search_specs.items():
        rs = RandomizedSearchCV(estimator=est, param_distributions=space, n_iter=n_iter,
                                scoring="f1_macro", cv=cv, random_state=seed, n_jobs=-1, refit=True)
        rs.fit(Xtr, ytr)
        best = rs.best_estimator_
        yhat = best.predict(Xte)
        acc = accuracy_score(yte, yhat)
        f1m = f1_score(yte, yhat, average="macro")
        prec, rec, _, _ = precision_recall_fscore_support(yte, yhat, average="macro", zero_division=0)
        rows.append({
            "Model": name, "BestParams": rs.best_params_, "CV5_F1_macro": float(rs.best_score_),
            "Test_Accuracy": float(acc), "Test_Precision_macro": float(prec),
            "Test_Recall_macro": float(rec), "Test_F1_macro": float(f1m)
        })
    return pd.DataFrame(rows).sort_values("Test_F1_macro", ascending=False)

# ----------------- main block -----------------
if __name__ == "__main__":
    os.makedirs(U_OUT_DIR, exist_ok=True)

    # 1) Load and filter rare classes so stratify + CV won’t fail
    df = pd.read_csv(U_DATA_PATH)
    counts = df[U_TARGET].value_counts()
    # keep only classes with at least 4 samples (safe for 80/20 split & CV>=2)
    keep_classes = counts[counts >= 4].index
    df = df[df[U_TARGET].isin(keep_classes)].copy()

    # 2) Stratified train/test split
    X = df.drop(columns=[U_TARGET])
    y = df[U_TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=U_TEST_SIZE, random_state=U_SEED, stratify=y)

    # 3) Pick CV folds based on the smallest class count in TRAIN
    min_train_class = ytr.value_counts().min()
    cv_splits = max(2, min(5, int(min_train_class)))  # ensure at least 2 folds; cap at 5
    # If a class still has only 1 sample in train (very rare), fallback to 2-fold without stratify is unsafe.
    # In that extreme case, drop that class or reduce test size and re-run. (Document in report if it happens.)

    # 4) Define models + spaces (A2 requirement)
    search_specs = U_build_search_space(seed=U_SEED)

    # 5) Run RandomizedSearchCV with safe cv_splits
    results = U_run_randomized_search(Xtr, ytr, Xte, yte, search_specs, seed=U_SEED, n_iter=25, n_splits=cv_splits)

    # 6) Save results table
    save_path = os.path.join(U_OUT_DIR, "Lab07_A2_results.csv")
    results.to_csv(save_path, index=False)

    # 7) Minimal console summary
    print("\n=== Lab07 A2: RandomizedSearchCV results ===")
    print(f"(cv folds used: {cv_splits})")
    print(results[["Model", "CV5_F1_macro", "Test_Accuracy", "Test_F1_macro"]])
    print(f"\nSaved CSV -> {save_path}")
