# ============================================================
# Lab 10 – A5: Explainability with LIME & SHAP (tabular)
# Dataset: C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv
# Target : 'class'
# Rules  : NO prints inside functions; only main prints/plots/saves.
# ============================================================

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# LIME & SHAP
from lime.lime_tabular import LimeTabularExplainer
import shap

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

# ---------------------- Training & Evaluation (no prints) ----------------------

def U_eval_classifier(model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "y_pred": y_pred
    }

# ---------------------- LIME helpers (no prints) ----------------------

def U_build_lime_explainer(X_train_scaled: np.ndarray, feature_names: List[str], class_names: List[str]) -> LimeTabularExplainer:
    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True,  # helps LIME on small/tabular sets
        sample_around_instance=True
    )
    return explainer

def U_lime_explain_instance(
    explainer: LimeTabularExplainer,
    model,
    x_scaled: np.ndarray,
    num_features: int = 5
) -> Any:
    # LIME expects a function returning class probabilities
    predict_proba = model.predict_proba
    exp = explainer.explain_instance(x_scaled, predict_proba, num_features=num_features)
    return exp

# ---------------------- SHAP helpers (no prints) ----------------------

def U_shap_tree_global(model: RandomForestClassifier, X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, feature_names: List[str], out_png: str) -> Dict[str, Any]:
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_scaled)  # list per class
    # Global summary (beeswarm) for all classes stacked
    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return {"explainer": explainer, "shap_values": shap_values}

def U_shap_linear_global(model: LogisticRegression, X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, feature_names: List[str], out_png: str) -> Dict[str, Any]:
    explainer = shap.LinearExplainer(model, X_train_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_scaled)  # array [n_samples, n_features]
    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return {"explainer": explainer, "shap_values": shap_values}

# ---------------------- Main (prints/saves only) ----------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    # ---- Config ----
    U_DATA_PATH     = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_TARGET_COL    = "class"
    DROP_IF_PRESENT = ["filename", "file", "filepath", "path", "id"]
    RNG = 42

    # 1) Load & numeric-only
    X_raw, y = U_load_csv(U_DATA_PATH, U_TARGET_COL)
    X, dropped_named, dropped_nonnum = U_numeric_only(X_raw, drop_if_present=DROP_IF_PRESENT)
    print(f"Loaded: {U_DATA_PATH} | X(before)={X_raw.shape} → X(numeric)={X.shape}, y={len(y)}")
    if dropped_named:  print("Dropped named columns:", dropped_named)
    if dropped_nonnum: print("Dropped non-numeric columns:", dropped_nonnum)

    # 2) Split + scale
    X_tr, X_te, y_tr, y_te, scaler, class_counts, feat_names = U_split_scale_safe(X, y, test_size=0.2, random_state=RNG)
    class_names = [str(c) for c in sorted(np.unique(y_tr))]
    print("Class counts:", class_counts)
    print(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}")

    # 3) Train models (same choices as A4)
    clf_rf  = RandomForestClassifier(n_estimators=150, random_state=RNG, class_weight="balanced_subsample")
    clf_log = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RNG, class_weight="balanced")

    rf_res  = U_eval_classifier(clf_rf,  X_tr, y_tr, X_te, y_te)
    log_res = U_eval_classifier(clf_log, X_tr, y_tr, X_te, y_te)

    print("\n=== Test Accuracy / Macro-F1 (for context) ===")
    print(f"RandomForest  → Acc={rf_res['accuracy']:.3f}, F1={rf_res['f1_macro']:.3f}")
    print(f"LogisticReg   → Acc={log_res['accuracy']:.3f}, F1={log_res['f1_macro']:.3f}")

    # 4) LIME (local explanation on 1–2 test points)
    out_dir = Path(U_DATA_PATH).parent
    lime_html_1 = out_dir / "lime_rf_instance0.html"
    lime_html_2 = out_dir / "lime_rf_instance1.html"

    lime_explainer = U_build_lime_explainer(X_tr, feat_names, class_names)
    # Pick first two test instances (if available)
    idxs = [0] + ([1] if len(X_te) > 1 else [])
    lime_summaries = []
    for i, idx in enumerate(idxs):
        exp = U_lime_explain_instance(lime_explainer, rf_res["model"], X_te[idx], num_features=min(5, X_tr.shape[1]))
        html_path = lime_html_1 if i == 0 else lime_html_2
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(exp.as_html())
        lime_summaries.append((idx, exp.as_list()))

    print("\n=== LIME (RandomForest) — local explanations saved ===")
    for idx, pairs in lime_summaries:
        print(f"- Test instance {idx}: top contributions:")
        for feat, weight in pairs[:5]:
            print(f"    {feat}: {weight:+.3f}")
    print(f"LIME HTML: {lime_html_1}")
    if len(idxs) > 1: print(f"LIME HTML: {lime_html_2}")

    # 5) SHAP — global explanations
    shap_png_rf  = out_dir / "shap_rf_summary.png"
    shap_png_log = out_dir / "shap_log_summary.png"

    shap_out_rf  = U_shap_tree_global(rf_res["model"], X_tr, X_te, feat_names, str(shap_png_rf))
    shap_out_log = U_shap_linear_global(log_res["model"], X_tr, X_te, feat_names, str(shap_png_log))

    print("\n=== SHAP — global summaries saved ===")
    print(f"RandomForest SHAP summary: {shap_png_rf}")
    print(f"LogisticReg  SHAP summary: {shap_png_log}")

    # 6) Brief guidance for report
    print("\nHow to interpret:")
    print("- LIME (local): explains a single prediction with a small linear surrogate; check the HTML for per-feature +/- contributions.")
    print("- SHAP (global): summary plot ranks features by average |SHAP| across the test set; color shows feature value impact on class probability.")
    print("- For your dataset, RF was strongest in A4; its SHAP summary reveals which raw features (e.g., 'mfcc1', 'rms') drive predictions overall.")
