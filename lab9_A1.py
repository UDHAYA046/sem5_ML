# ============================================================
# Lab 9 - A1: Stacking Classifier (Windows-safe)
# Author : S. Udhaya Sankari (BL.EN.U4CSE23150)
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


U_OUTDIR = r"C:\Users\Udhaya\sem5_ML\lab9_output_figures"

def U_make_outdir(p): Path(p).mkdir(parents=True, exist_ok=True); return p
def U_now(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def U_print_banner():
    print("\n========================================")
    print(" Lab 9 - A1 : Stacking Classifier")
    print(" Student : S. Udhaya Sankari | Roll : BL.EN.U4CSE23150")
    print("========================================\n")

def U_load_data():
    from sklearn.datasets import load_iris
    d = load_iris()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target, name="target")
    return X, y

def U_build_stacking():
    base = [
        ('U_lr',  LogisticRegression(max_iter=2000)),
        ('U_rf',  RandomForestClassifier(n_estimators=120, random_state=42)),
        ('U_svm', SVC(kernel='linear', probability=True))
    ]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=2000),
        stack_method='predict_proba',
        cv=cv,
        n_jobs=None  # Windows-safe
    )
    return model

def U_build_pipeline(model):
    return Pipeline([('U_scaler', StandardScaler()), ('U_clf', model)])

def U_eval_and_save(y_true, y_pred, name, outdir):
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)

    txt = f"Model: {name}\nAccuracy: {acc:.4f}\n\n{rep}\nConfusion Matrix:\n{cm}\n"
    fp_txt = os.path.join(outdir, f"A1_{name}_metrics_{U_now()}.txt")
    with open(fp_txt, "w", encoding="utf-8") as f: f.write(txt)
    print(f"[Saved] {fp_txt}")

    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"A1 Confusion: {name}"); plt.colorbar()
    t = np.arange(len(np.unique(y_true)))
    plt.xticks(t, t); plt.yticks(t, t)
    plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout()
    fp_png = os.path.join(outdir, f"A1_{name}_confusion_{U_now()}.png")
    plt.savefig(fp_png, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[Saved] {fp_png}")

def U_main():
    U_print_banner()
    outdir = U_make_outdir(U_OUTDIR)
    X, y = U_load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = U_build_stacking()
    pipe  = U_build_pipeline(model)

    print("[INFO] Fitting Stacking (CV=5, no parallel)...")
    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)
    U_eval_and_save(yte, ypred, "StackingClassifier", outdir)
    print("[DONE] A1 complete.")

if __name__ == "__main__":
    U_main()
