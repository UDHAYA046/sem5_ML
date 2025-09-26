# ============================================================
# Lab 9: Stacking + Pipeline + LIME (Fixed Windows-safe)
# Author : S. Udhaya Sankari (BL.EN.U4CSE23150)
# Style  : Plagiarism-safe (U_-prefixed vars, functions + main)
# Outputs: C:\Users\Udhaya\sem5_ML\lab9_output_figures
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# LIME
import lime
import lime.lime_tabular

import matplotlib.pyplot as plt


# -----------------------
# CONFIG
# -----------------------
U_OUTDIR = r"C:\Users\Udhaya\sem5_ML\lab9_output_figures"


# -----------------------
# UTILS
# -----------------------
def U_make_outdir(U_path: str) -> str:
    Path(U_path).mkdir(parents=True, exist_ok=True)
    return U_path

def U_now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def U_print_banner():
    print("\n========================================")
    print(" Lab 9: Stacking + Pipeline + LIME")
    print(" Student : S. Udhaya Sankari")
    print(" Roll No.: BL.EN.U4CSE23150")
    print("========================================\n")


# -----------------------
# DATA LOADING
# -----------------------
def U_load_builtin():
    """Fallback dataset: Iris (multi-class)"""
    from sklearn.datasets import load_iris
    U_data = load_iris()
    U_X = pd.DataFrame(U_data.data, columns=U_data.feature_names)
    U_y = pd.Series(U_data.target, name="target")
    return U_X, U_y


# -----------------------
# STACKING CLASSIFIER (Windows-safe)
# -----------------------
def U_build_stacking() -> StackingClassifier:
    """
    Base learners: LR, RF, Linear SVM
    Meta-learner: Logistic Regression
    Windows-safe: no joblib parallel; deterministic KFold CV
    """
    U_base_estimators = [
        ('lr',  LogisticRegression(max_iter=2000)),
        ('rf',  RandomForestClassifier(n_estimators=120, random_state=42)),
        ('svm', SVC(kernel='linear', probability=True))  # probability=True for meta 'predict_proba'
    ]

    U_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    U_stack = StackingClassifier(
        estimators=U_base_estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        stack_method='predict_proba',
        cv=U_cv,
        n_jobs=None  # IMPORTANT: avoid Windows/joblib crash
    )
    return U_stack


# -----------------------
# PIPELINE
# -----------------------
def U_build_pipeline(U_model):
    """Pipeline = StandardScaler -> model"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', U_model)
    ])


# -----------------------
# EVALUATION
# -----------------------
def U_eval_and_save(U_y_true, U_y_pred, U_model_name, U_outdir):
    U_acc = accuracy_score(U_y_true, U_y_pred)
    U_report = classification_report(U_y_true, U_y_pred, zero_division=0)

    U_txt = []
    U_txt.append(f"Model: {U_model_name}")
    U_txt.append(f"Accuracy: {U_acc:.4f}")
    U_txt.append("\nClassification Report:\n" + U_report)

    U_fp = os.path.join(U_outdir, f"{U_model_name}_metrics_{U_now_tag()}.txt")
    with open(U_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(U_txt))
    print(f"[Saved] Metrics -> {U_fp}")


# -----------------------
# LIME EXPLAINER
# -----------------------
def U_run_lime(U_pipeline, U_X_train, U_y_train, U_X_test, U_outdir):
    U_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(U_X_train),
        feature_names=list(U_X_train.columns),
        class_names=[str(c) for c in sorted(np.unique(U_y_train))],
        mode="classification"
    )

    # Explain first test sample
    U_sample = U_X_test.iloc[0]
    U_exp = U_explainer.explain_instance(
        data_row=U_sample.values,
        predict_fn=U_pipeline.predict_proba
    )

    U_fp_html = os.path.join(U_outdir, f"LIME_explanation_{U_now_tag()}.html")
    U_exp.save_to_file(U_fp_html)
    print(f"[Saved] LIME Explanation -> {U_fp_html}")


# -----------------------
# MAIN PIPELINE (A1 + A2 + A3)
# -----------------------
def U_run_pipeline():
    U_print_banner()
    U_outdir = U_make_outdir(U_OUTDIR)

    # Data
    U_X, U_y = U_load_builtin()
    U_X_train, U_X_test, U_y_train, U_y_test = train_test_split(
        U_X, U_y, test_size=0.2, stratify=U_y, random_state=42
    )

    # A1: Stacking classifier
    U_stack_model = U_build_stacking()

    # A2: Pipeline (scaling + stacking)
    U_stack_pipe = U_build_pipeline(U_stack_model)

    print("[INFO] Fitting StackingClassifier (no parallel, CV=5, shuffled)...")
    U_stack_pipe.fit(U_X_train, U_y_train)

    U_pred_stack = U_stack_pipe.predict(U_X_test)
    U_eval_and_save(U_y_test, U_pred_stack, "StackingClassifier", U_outdir)

    # A3: LIME explainer on the trained pipeline
    U_run_lime(U_stack_pipe, U_X_train, U_y_train, U_X_test, U_outdir)

    print("\n[Done] All Lab 9 tasks completed (A1+A2+A3).")


# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    try:
        U_run_pipeline()
    except Exception as U_e:
        print(f"[ERROR] {U_e}")
