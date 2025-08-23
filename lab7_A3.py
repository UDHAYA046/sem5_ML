# U_Lab07_A3_plus.py
# A3-Plus: Default models + Confusion Matrices + One-vs-Rest ROC curves (saved to disk)
# Same U_ rules as above.

import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------- CONFIG ----------
U_DATA_PATH = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
U_OUT_DIR   = r"C:\Users\Udhaya\sem5_ML\lab7_output_figures"
U_TARGET    = "class"
U_TEST_SIZE = 0.30
U_SEED      = 42
# ----------------------------

# ---------- HELPERS ----------
def U_load_numeric_xy(csv_path: str, target_col: str):
    U_df = pd.read_csv(csv_path)
    U_y = U_df[target_col]
    U_X = U_df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
    U_X = U_X.replace([np.inf, -np.inf], np.nan).fillna(U_X.mean(numeric_only=True))
    return U_X, U_y

def U_safe_split(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    U_can = (y.value_counts().min() >= 2)
    U_Xtr, U_Xte, U_ytr, U_yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if U_can else None
    )
    return U_Xtr, U_Xte, U_ytr, U_yte, np.unique(y)

def U_default_models(seed: int):
    return {
        "KNN": Pipeline([("U_scale", StandardScaler()), ("U_clf", KNeighborsClassifier())]),
        "SVM": Pipeline([("U_scale", StandardScaler()), ("U_clf", SVC(probability=True, random_state=seed))]),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(random_state=seed),
    }

def U_save_confusion(y_true, y_pred, labels, name: str, out_dir: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    p = os.path.join(out_dir, f"Lab07_A3_CM_{name}.png")
    plt.savefig(p, dpi=200); plt.close()
    return p

def U_save_roc(model, X_test: pd.DataFrame, y_test: pd.Series, labels, name: str, out_dir: str):
    y_bin = label_binarize(y_test, classes=labels)
    n_classes = y_bin.shape[1]
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)
    if y_score.ndim == 1:  # binary decision_function case
        y_score = y_score.reshape(-1, 1)

    plt.figure(figsize=(7,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {labels[i]} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.title(f"ROC (OvR) – {name}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right"); plt.tight_layout()
    p = os.path.join(out_dir, f"Lab07_A3_ROC_{name}.png")
    plt.savefig(p, dpi=200); plt.close()
    return p
# ------------------------------------------

# --------------------------- MAIN ------------------------
if __name__ == "__main__":
    os.makedirs(U_OUT_DIR, exist_ok=True)

    U_X, U_y = U_load_numeric_xy(U_DATA_PATH, U_TARGET)
    U_Xtr, U_Xte, U_ytr, U_yte, U_labels = U_safe_split(U_X, U_y, U_TEST_SIZE, U_SEED)

    U_models = U_default_models(U_SEED)
    U_rows = []

    for U_name, U_model in U_models.items():
        U_model.fit(U_Xtr, U_ytr)
        U_pred = U_model.predict(U_Xte)
        U_acc  = accuracy_score(U_yte, U_pred)
        U_f1   = f1_score(U_yte, U_pred, average="weighted")
        U_rows.append({"Model": U_name, "Accuracy": U_acc, "F1_weighted": U_f1})
        U_save_confusion(U_yte, U_pred, U_labels, U_name, U_OUT_DIR)
        U_save_roc(U_model, U_Xte, U_yte, U_labels, U_name, U_OUT_DIR)

    U_df = pd.DataFrame(U_rows).sort_values("F1_weighted", ascending=False)
    U_csv = os.path.join(U_OUT_DIR, "Lab07_A3_plus_default_results.csv")
    U_df.to_csv(U_csv, index=False)
    print("Saved A3-plus table ->", U_csv)
