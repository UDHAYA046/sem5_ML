# ============================================================
# Lab 9: Stacking, Pipelines, and LIME
# Author: S. Udhaya Sankari (BL.EN.U4CSE23150)
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

# For explanations
import lime
import lime.lime_tabular


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
    """Fallback dataset: Iris"""
    from sklearn.datasets import load_iris
    U_data = load_iris()
    U_X = pd.DataFrame(U_data.data, columns=U_data.feature_names)
    U_y = pd.Series(U_data.target, name="target")
    return U_X, U_y


# -----------------------
# STACKING CLASSIFIER
# -----------------------
def U_build_stacking():
    """Define stacking classifier with multiple base learners"""
    U_base_estimators = [
        ('lr', LogisticRegression(max_iter=2000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ]
    U_final_est = LogisticRegression(max_iter=2000)
    U_stack = StackingClassifier(
        estimators=U_base_estimators,
        final_estimator=U_final_est,
        cv=5,
        n_jobs=-1
    )
    return U_stack


# -----------------------
# PIPELINE
# -----------------------
def U_build_pipeline(U_model):
    """Pipeline: scaling + model"""
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
        feature_names=U_X_train.columns,
        class_names=[str(c) for c in np.unique(U_y_train)],
        mode="classification"
    )

    # Pick one sample to explain
    U_sample = U_X_test.iloc[0]
    U_exp = U_explainer.explain_instance(
        data_row=U_sample,
        predict_fn=U_pipeline.predict_proba
    )

    U_fp_html = os.path.join(U_outdir, f"LIME_explanation_{U_now_tag()}.html")
    U_exp.save_to_file(U_fp_html)
    print(f"[Saved] LIME Explanation -> {U_fp_html}")


# -----------------------
# MAIN
# -----------------------
def U_run_pipeline():
    U_print_banner()
    U_outdir = U_make_outdir(U_OUTDIR)

    # Load dataset
    U_X, U_y = U_load_builtin()
    U_X_train, U_X_test, U_y_train, U_y_test = train_test_split(
        U_X, U_y, test_size=0.2, stratify=U_y, random_state=42
    )

    # A1: Stacking Classifier
    U_stack_model = U_build_stacking()
    U_stack_pipe = U_build_pipeline(U_stack_model)
    U_stack_pipe.fit(U_X_train, U_y_train)
    U_pred_stack = U_stack_pipe.predict(U_X_test)
    U_eval_and_save(U_y_test, U_pred_stack, "StackingClassifier", U_outdir)

    # A2: Example pipeline already shown (scaler + stacking)
    print("[INFO] Pipeline executed: Scaling -> StackingClassifier")

    # A3: LIME Explanation
    U_run_lime(U_stack_pipe, U_X_train, U_y_train, U_X_test, U_outdir)


if __name__ == "__main__":
    U_run_pipeline()
