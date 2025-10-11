# ============================================================
# Lab 10 – A4 (final): Sequential Feature Selection / Reduction + Comparison
# Author: S. Udhaya Sankari
# Rules: functions have NO prints; prints/plots only in main.
# Fixes: SFS uses KFold(2) + n_jobs=1; no 'strict=' in zip (Py3.9 safe)
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector, RFE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.base import clone
import warnings


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, Dict[Any, int], List[str]]:
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    feature_names = X.columns.tolist()

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
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, counts, feature_names


# ---------------------- PCA (no prints) ----------------------

def U_apply_pca(X_train: np.ndarray, X_test: np.ndarray, variance_retained: float, rng: int = 42
               ) -> Tuple[np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=variance_retained, random_state=rng)
    return pca.fit_transform(X_train), pca.transform(X_test), pca


# ---------------------- SFS / RFE utilities (no prints) ----------------------

def U_eval_subset(estimator, Xtr_sel, ytr, Xte_sel, yte) -> Dict[str, Any]:
    model = clone(estimator)
    model.fit(Xtr_sel, ytr)
    y_pred = model.predict(Xte_sel)
    return {
        "accuracy": accuracy_score(yte, y_pred),
        "f1_macro": f1_score(yte, y_pred, average="macro", zero_division=0),
        "y_pred": y_pred,
    }

def U_run_sfs(
    base_estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    direction: str = "forward",
    k_values: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], Dict[int, List[str]]]:
    """
    SFS with KFold(2) and n_jobs=1 (stable on Windows; tolerant to single-sample class).
    Select best k by macro-F1 (tie-break: accuracy).
    """
    if k_values is None:
        k_values = list(range(2, max(3, X_train.shape[1]) + 1))

    selected_by_k: Dict[int, List[str]] = {}
    best = {"k": None, "accuracy": -1.0, "f1_macro": -1.0, "mask": None}

    cv2 = KFold(n_splits=2, shuffle=True, random_state=42)

    for k in k_values:
        sfs = SequentialFeatureSelector(
            base_estimator,
            n_features_to_select=k,
            direction=direction,
            cv=cv2,
            n_jobs=1,            # key fix for Windows/joblib
            scoring="f1_macro",
        )
        sfs.fit(X_train, y_train)
        mask = sfs.get_support()
        Xtr_sel = X_train[:, mask]
        Xte_sel = X_test[:, mask]
        sel_names = [f for f, keep in zip(feature_names, mask.tolist()) if keep]
        selected_by_k[k] = sel_names

        res = U_eval_subset(base_estimator, Xtr_sel, y_train, Xte_sel, y_test)
        if (res["f1_macro"] > best["f1_macro"]) or (np.isclose(res["f1_macro"], best["f1_macro"]) and res["accuracy"] > best["accuracy"]):
            best = {"k": k, "accuracy": res["accuracy"], "f1_macro": res["f1_macro"], "mask": mask}

    return best, selected_by_k

def U_run_rfe(
    base_estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    k_values: Optional[List[int]] = None
) -> Tuple[Dict[str, Any], Dict[int, List[str]]]:
    if k_values is None:
        k_values = list(range(2, max(3, X_train.shape[1]) + 1))

    selected_by_k: Dict[int, List[str]] = {}
    best = {"k": None, "accuracy": -1.0, "f1_macro": -1.0, "mask": None}

    for k in k_values:
        rfe = RFE(estimator=base_estimator, n_features_to_select=k, step=1)
        rfe.fit(X_train, y_train)
        mask = rfe.support_
        Xtr_sel = X_train[:, mask]
        Xte_sel = X_test[:, mask]
        sel_names = [f for f, keep in zip(feature_names, mask.tolist()) if keep]
        selected_by_k[k] = sel_names

        res = U_eval_subset(base_estimator, Xtr_sel, y_train, Xte_sel, y_test)
        if (res["f1_macro"] > best["f1_macro"]) or (np.isclose(res["f1_macro"], best["f1_macro"]) and res["accuracy"] > best["accuracy"]):
            best = {"k": k, "accuracy": res["accuracy"], "f1_macro": res["f1_macro"], "mask": mask}

    return best, selected_by_k


# ---------------------- Baseline evaluation (no prints) ----------------------

def U_eval_classifier(model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    m = clone(model)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "y_pred": y_pred
    }


# ---------------------- Main (prints only here) ----------------------

if __name__ == "__main__":
    # Optional: quiet known small-sample warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # ---- Config ----
    U_DATA_PATH    = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_TARGET_COL   = "class"
    DROP_IF_PRESENT = ["filename", "file", "filepath", "path", "id"]
    RNG = 42

    # 1) Load & numeric-only
    X_raw, y = U_load_csv(U_DATA_PATH, U_TARGET_COL)
    X, dropped_named, dropped_nonnum = U_numeric_only(X_raw, drop_if_present=DROP_IF_PRESENT)
    print(f"Loaded: {U_DATA_PATH} | X(before)={X_raw.shape} → X(numeric)={X.shape}, y={len(y)}")
    if dropped_named:  print("Dropped named columns:", dropped_named)
    if dropped_nonnum: print("Dropped non-numeric columns:", dropped_nonnum)

    # 2) Robust split + scaling
    X_tr, X_te, y_tr, y_te, scaler, class_counts, feat_names = U_split_scale_safe(X, y, test_size=0.2, random_state=RNG)
    print("Class counts:", class_counts)
    print(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
    p = X_tr.shape[1]
    k_list = list(range(2, max(2, p) + 1))

    # 3) Models (class-weighted for imbalance)
    clf_log = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RNG, class_weight="balanced")
    clf_rf  = RandomForestClassifier(n_estimators=150, random_state=RNG, class_weight="balanced_subsample")

    # 4) Baseline
    base_log = U_eval_classifier(clf_log, X_tr, y_tr, X_te, y_te)
    base_rf  = U_eval_classifier(clf_rf,  X_tr, y_tr, X_te, y_te)

    # 5) PCA-99 and PCA-95 (to compare with A2/A3)
    X_tr_p99, X_te_p99, pca99 = U_apply_pca(X_tr, X_te, variance_retained=0.99, rng=RNG)
    X_tr_p95, X_te_p95, pca95 = U_apply_pca(X_tr, X_te, variance_retained=0.95, rng=RNG)
    res_p99_log = U_eval_classifier(clf_log, X_tr_p99, y_tr, X_te_p99, y_te)
    res_p99_rf  = U_eval_classifier(clf_rf,  X_tr_p99, y_tr, X_te_p99, y_te)
    res_p95_log = U_eval_classifier(clf_log, X_tr_p95, y_tr, X_te_p95, y_te)
    res_p95_rf  = U_eval_classifier(clf_rf,  X_tr_p95, y_tr, X_te_p95, y_te)

    # 6) SFS (forward/backward) + RFE for both models (stable settings)
    best_sfs_f_log, map_sfs_f_log = U_run_sfs(clf_log, X_tr, y_tr, X_te, y_te, feat_names, direction="forward",  k_values=k_list)
    best_sfs_b_log, map_sfs_b_log = U_run_sfs(clf_log, X_tr, y_tr, X_te, y_te, feat_names, direction="backward", k_values=k_list)
    best_rfe_log,    map_rfe_log  = U_run_rfe(clf_log,  X_tr, y_tr, X_te, y_te, feat_names, k_values=k_list)

    best_sfs_f_rf, map_sfs_f_rf = U_run_sfs(clf_rf, X_tr, y_tr, X_te, y_te, feat_names, direction="forward",  k_values=k_list)
    best_sfs_b_rf, map_sfs_b_rf = U_run_sfs(clf_rf, X_tr, y_tr, X_te, y_te, feat_names, direction="backward", k_values=k_list)
    best_rfe_rf,   map_rfe_rf   = U_run_rfe(clf_rf,  X_tr, y_tr, X_te, y_te, feat_names, k_values=k_list)

    # 7) Comparison table
    print("\n=== A4 Comparison: Baseline vs PCA vs SFS/RFE (Test set) ===")
    print(f"{'Model / Method':<28} | {'k/comp':<7} | {'Acc':<6} | {'F1m':<6}")
    print("-"*60)
    def row(name, kval, res): print(f"{name:<28} | {kval:<7} | {res['accuracy']:<6.3f} | {res['f1_macro']:<6.3f}")

    row("LogReg – Baseline", p, base_log)
    row("LogReg – PCA 99%", X_tr_p99.shape[1], res_p99_log)
    row("LogReg – PCA 95%", X_tr_p95.shape[1], res_p95_log)
    row("LogReg – SFS Forward",  best_sfs_f_log['k'], {"accuracy": best_sfs_f_log["accuracy"], "f1_macro": best_sfs_f_log["f1_macro"]})
    row("LogReg – SFS Backward", best_sfs_b_log['k'], {"accuracy": best_sfs_b_log["accuracy"], "f1_macro": best_sfs_b_log["f1_macro"]})
    row("LogReg – RFE",          best_rfe_log['k'],   {"accuracy": best_rfe_log["accuracy"],   "f1_macro": best_rfe_log["f1_macro"]})

    print("-"*60)
    row("RF – Baseline", p, base_rf)
    row("RF – PCA 99%", X_tr_p99.shape[1], res_p99_rf)
    row("RF – PCA 95%", X_tr_p95.shape[1], res_p95_rf)
    row("RF – SFS Forward",  best_sfs_f_rf['k'], {"accuracy": best_sfs_f_rf["accuracy"], "f1_macro": best_sfs_f_rf["f1_macro"]})
    row("RF – SFS Backward", best_sfs_b_rf['k'], {"accuracy": best_sfs_b_rf["accuracy"], "f1_macro": best_sfs_b_rf["f1_macro"]})
    row("RF – RFE",          best_rfe_rf['k'],   {"accuracy": best_rfe_rf["accuracy"],   "f1_macro": best_rfe_rf["f1_macro"]})

    # 8) Show selected subsets (names)
    def picked_names(best_mask, names):
        return [f for f, keep in zip(names, best_mask.tolist()) if keep]

    print("\nSelected subsets:")
    print("• LogReg SFS-Forward:",  picked_names(best_sfs_f_log["mask"], feat_names),  f"(k={best_sfs_f_log['k']})")
    print("• LogReg SFS-Backward:", picked_names(best_sfs_b_log["mask"], feat_names),  f"(k={best_sfs_b_log['k']})")
    print("• LogReg RFE:",          picked_names(best_rfe_log["mask"],    feat_names), f"(k={best_rfe_log['k']})")
    print("• RF SFS-Forward:",      picked_names(best_sfs_f_rf["mask"],   feat_names), f"(k={best_sfs_f_rf['k']})")
    print("• RF SFS-Backward:",     picked_names(best_sfs_b_rf["mask"],   feat_names), f"(k={best_sfs_b_rf['k']})")
    print("• RF RFE:",              picked_names(best_rfe_rf["mask"],     feat_names), f"(k={best_rfe_rf['k']})")

    print("\nNotes:")
    print("- SFS uses KFold(2) and single-core to avoid worker termination and to tolerate a class with a single sample.")
    print("- Selection is supervised by macro-F1; results are directly comparable to A2 (PCA-99%) and A3 (PCA-95%).")
