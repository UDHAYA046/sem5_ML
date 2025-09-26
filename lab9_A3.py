# ============================================================
# Lab 9 - A3: LIME Explainer on Pipeline
# Author : S. Udhaya Sankari (BL.EN.U4CSE23150)
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import lime
import lime.lime_tabular

U_OUTDIR = r"C:\Users\Udhaya\sem5_ML\lab9_output_figures"

def U_make_outdir(p): Path(p).mkdir(parents=True, exist_ok=True); return p
def U_now(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def U_print_banner():
    print("\n========================================")
    print(" Lab 9 - A3 : LIME on Pipeline")
    print(" Student : S. Udhaya Sankari | Roll : BL.EN.U4CSE23150")
    print("========================================\n")

def U_load_data():
    from sklearn.datasets import load_iris
    d = load_iris()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target, name="target")
    return X, y

def U_build_pipeline():
    return Pipeline([
        ('U_scaler', StandardScaler()),
        ('U_clf',   LogisticRegression(max_iter=2000))
    ])

def U_run_lime(trained_pipeline, X_train, y_train, X_test, outdir):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=list(X_train.columns),
        class_names=[str(c) for c in sorted(np.unique(y_train))],
        mode="classification"
    )
    x0 = X_test.iloc[0]
    exp = explainer.explain_instance(
        data_row=x0.values,
        predict_fn=trained_pipeline.predict_proba
    )
    fp = os.path.join(outdir, f"A3_LIME_explanation_{U_now()}.html")
    exp.save_to_file(fp)
    print(f"[Saved] {fp}")

def U_main():
    U_print_banner()
    outdir = U_make_outdir(U_OUTDIR)
    X, y = U_load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = U_build_pipeline()
    print("[INFO] Fitting pipeline for LIME...")
    pipe.fit(Xtr, ytr)
    U_run_lime(pipe, Xtr, ytr, Xte, outdir)
    print("[DONE] A3 complete (open the HTML in a browser).")

if __name__ == "__main__":
    U_main()
