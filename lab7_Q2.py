# U_Lab07_O2_LIME.py
# O2: LIME local explanations for a RandomForest classifier (tabular)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ------------------- CONFIG -------------------
U_DATA_PATH = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
U_OUT_DIR   = r"C:\Users\Udhaya\sem5_ML\lab7_output_figures"
U_TARGET    = "class"
U_TEST_SIZE = 0.30
U_SEED      = 42
U_INSTANCE_INDEX = 0   # which test sample to explain (0-based)
# ----------------------------------------------


# ----------------- FUNCTIONS (no prints) -----------------
def U_load_numeric_xy(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean(numeric_only=True))
    return X, y

def U_filter_rare_classes(X: pd.DataFrame, y: pd.Series, min_count: int = 2):
    counts = y.value_counts()
    valid = counts[counts >= min_count].index
    mask = y.isin(valid)
    return X[mask], y[mask]

def U_train_rf(Xtr, ytr):
    rf = RandomForestClassifier(random_state=U_SEED)
    rf.fit(Xtr, ytr)
    return rf

def U_save_lime_explanation(explainer, model, X_train: pd.DataFrame,
                            X_test: pd.DataFrame, idx: int, out_png: str):
    """Generate and save a LIME explanation for X_test.iloc[idx]."""
    exp = explainer.explain_instance(
        data_row=X_test.iloc[idx].values,     # pass raw values
        predict_fn=model.predict_proba,
        num_features=min(10, X_test.shape[1])
    )
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png
# ---------------------------------------------------------


# --------------------------- MAIN ------------------------
if __name__ == "__main__":
    os.makedirs(U_OUT_DIR, exist_ok=True)

    # 1) Load numeric X, y and filter rare labels
    U_X, U_y = U_load_numeric_xy(U_DATA_PATH, U_TARGET)
    U_X, U_y = U_filter_rare_classes(U_X, U_y, min_count=2)

    # 2) Safe split (try stratify else plain)
    try:
        U_Xtr, U_Xte, U_ytr, U_yte = train_test_split(U_X, U_y, test_size=U_TEST_SIZE,
                                                      random_state=U_SEED, stratify=U_y)
    except ValueError:
        U_Xtr, U_Xte, U_ytr, U_yte = train_test_split(U_X, U_y, test_size=U_TEST_SIZE,
                                                      random_state=U_SEED)

    # 3) Train RandomForest (LIME just needs a classifier with predict_proba)
    U_rf = U_train_rf(U_Xtr, U_ytr)

    # 4) Build LIME explainer (class names from training labels)
    U_class_names = [str(c) for c in sorted(U_ytr.unique())]
    U_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=U_Xtr.values,
        feature_names=U_Xtr.columns.tolist(),
        class_names=U_class_names,
        mode="classification"
    )

    # 5) Save explanation for a chosen test instance
    U_idx = int(np.clip(U_INSTANCE_INDEX, 0, len(U_Xte) - 1))
    U_png = os.path.join(U_OUT_DIR, f"Lab07_O2_LIME_instance{U_idx}.png")
    U_save_lime_explanation(U_explainer, U_rf, U_Xtr, U_Xte, U_idx, U_png)

    print(f"LIME explanation saved -> {U_png}")
