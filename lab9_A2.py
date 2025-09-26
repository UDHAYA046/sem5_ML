# ============================================================
# Lab 9 - A2: Parallel Feature Processing Pipeline
# Author : S. Udhaya Sankari (BL.EN.U4CSE23150)
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

U_OUTDIR = r"C:\Users\Udhaya\sem5_ML\lab9_output_figures"

def U_make_outdir(p): Path(p).mkdir(parents=True, exist_ok=True); return p
def U_now(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def U_print_banner():
    print("\n========================================")
    print(" Lab 9 - A2 : Parallel Pipeline (FeatureUnion)")
    print(" Student : S. Udhaya Sankari | Roll : BL.EN.U4CSE23150")
    print("========================================\n")

def U_load_data():
    from sklearn.datasets import load_iris
    d = load_iris()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target, name="target")
    return X, y

def U_build_parallel_union():
    # Branch 1: Scaled full features
    U_branch_scaled = Pipeline([
        ('U_scaler', StandardScaler())
    ])
    # Branch 2: Scaled + PCA(2) features
    U_branch_pca = Pipeline([
        ('U_scaler2', StandardScaler()),
        ('U_pca2', PCA(n_components=2, random_state=42))
    ])
    # FeatureUnion runs branches simultaneously and concatenates outputs
    U_union = FeatureUnion([
        ('U_scaled_all', U_branch_scaled),
        ('U_scaled_pca2', U_branch_pca)
    ])
    return U_union

def U_build_pipeline():
    U_union = U_build_parallel_union()
    U_clf   = LogisticRegression(max_iter=2000)
    return Pipeline([
        ('U_features', U_union),
        ('U_clf', U_clf)
    ])

def U_eval_and_save(y_true, y_pred, outdir):
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, zero_division=0)
    fp = os.path.join(outdir, f"A2_Pipeline_metrics_{U_now()}.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(f"A2 Parallel Pipeline\nAccuracy: {acc:.4f}\n\n{rep}\n")
    print(f"[Saved] {fp}")

def U_main():
    U_print_banner()
    outdir = U_make_outdir(U_OUTDIR)
    X, y = U_load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    U_pipe = U_build_pipeline()
    print("[INFO] Fitting parallel FeatureUnion pipeline...")
    U_pipe.fit(Xtr, ytr)
    ypred = U_pipe.predict(Xte)
    U_eval_and_save(yte, ypred, outdir)
    print("[DONE] A2 complete.")

if __name__ == "__main__":
    U_main()
