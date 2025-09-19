# ------------------------------------------------------------
# Lab 8 – A12 | Use scikit-learn MLPClassifier on project data
# Author : S. Udhaya Sankari | BL.EN.U4CSE23150
#
# What it does:
#  - Load CSV dataset
#  - Auto-detect target column (label/target/class/y/outcome, or low-unique col)
#  - Preprocess (StandardScaler for numeric, OneHotEncoder for categoricals)
#  - GridSearchCV to pick a decent MLP
#  - Train/test split (stratified if possible), evaluate, and save artifacts:
#    * confusion_matrix.png
#    * learning_curve.png
#    * classification_report.txt
#    * summary.json
#    * best_model.joblib (optional)
# ------------------------------------------------------------

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Optional: save the fitted pipeline for later use
try:
    import joblib
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

warnings.filterwarnings("ignore")

# --------------------------- CONFIG ---------------------------
# Point this to your dataset. If left as relative, it will look in the same folder.
DATA_PATH = "features_lab3_labeled.csv"

# Where to save plots & reports
OUT_DIR = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", "lab8_output_figures")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 123

# ---------------------- HELPER FUNCTIONS ----------------------
def guess_target_column(df: pd.DataFrame) -> str:
    """Heuristic to infer the target column."""
    name_hits = [c for c in df.columns if c.lower() in ("label", "target", "class", "y", "outcome")]
    if name_hits:
        return name_hits[0]
    # otherwise pick a low-cardinality column (but at least 2 classes)
    nrows = len(df)
    small_uniques = [(c, df[c].nunique()) for c in df.columns]
    small_uniques.sort(key=lambda x: x[1])
    for c, u in small_uniques:
        if 2 <= u <= max(10, int(0.05 * nrows)):
            return c
    # last-resort: last column
    return df.columns[-1]

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric/categorical features."""
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        # use sparse=False for broad sklearn compatibility
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols))
    if not transformers:
        # edge case: no features? (unlikely)
        raise ValueError("No feature columns detected.")
    return ColumnTransformer(transformers=transformers, remainder="drop")

# ----------------------------- MAIN ---------------------------
def main():
    # ---- Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # ---- Detect target
    target_col = guess_target_column(df)
    X = df.drop(columns=[target_col])
    y_raw = df[target_col].copy()

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    # ---- Preprocessing
    preprocessor = build_preprocessor(X)

    # ---- Base estimator (to be tuned)
    base_mlp = MLPClassifier(max_iter=300, random_state=RANDOM_STATE)

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", base_mlp)])

    # ---- Small grid for quick tuning (expand if you want)
    param_grid = {
        "clf__hidden_layer_sizes": [(16,), (32,), (16, 8)],
        "clf__activation": ["relu", "logistic"],
        "clf__solver": ["adam"],
        "clf__learning_rate_init": [0.001, 0.01],
        "clf__alpha": [0.0001, 0.001],
    }

    # ---- Train/Test split (stratify only if every class has >= 2 samples)
    uniques, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    stratify_y = y if min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y, shuffle=True
    )

    # ---- Cross-validation strategy
    if min_count >= 2:
        n_splits = max(2, int(min(5, min_count)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # ---- Grid search
    grid = GridSearchCV(
        pipe, param_grid, cv=cv, n_jobs=None, scoring="accuracy", refit=True, verbose=0
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    cv_best_acc = grid.best_score_

    # ---- Evaluate
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # Ensure report includes all classes even if absent in test set
    all_label_ids = np.arange(len(class_names))
    report = classification_report(
        y_test, y_pred, labels=all_label_ids, target_names=class_names, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=all_label_ids)

    # ---- Confusion matrix plot
    fig = plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(values_format="d")
    plt.title("A12 – Confusion Matrix (MLPClassifier)")
    cm_path = os.path.join(OUT_DIR, "U_A12_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train, cv=cv,
        train_sizes=np.linspace(0.25, 1.0, 5), scoring="accuracy"
    )
    fig2 = plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="CV")
    plt.title("A12 – Learning Curve (MLPClassifier)")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.6)
    lc_path = os.path.join(OUT_DIR, "U_A12_learning_curve.png")
    plt.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ---- Save text & JSON summaries
    with open(os.path.join(OUT_DIR, "U_A12_classification_report.txt"), "w") as f:
        f.write(report)

    summary = {
        "data_path": os.path.abspath(DATA_PATH),
        "target_column": target_col,
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "classes": class_names,
        "cv_strategy": type(cv).__name__,
        "best_params": best_params,
        "cv_best_accuracy": float(cv_best_acc),
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "artifacts": {
            "confusion_matrix_png": os.path.abspath(cm_path),
            "learning_curve_png": os.path.abspath(lc_path),
            "classification_report_txt": os.path.abspath(os.path.join(OUT_DIR, "U_A12_classification_report.txt")),
        },
    }
    with open(os.path.join(OUT_DIR, "U_A12_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- (Optional) Save the trained pipeline
    if _HAVE_JOBLIB:
        model_path = os.path.join(OUT_DIR, "U_A12_best_model.joblib")
        joblib.dump(best_model, model_path)
        summary["artifacts"]["best_model_joblib"] = os.path.abspath(model_path)
        with open(os.path.join(OUT_DIR, "U_A12_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ---- Console summary
    print("\n=== A12: MLPClassifier on project dataset ===")
    print(f"Data file             : {os.path.abspath(DATA_PATH)}")
    print(f"Target column         : {target_col}")
    print(f"Samples / Features    : {len(df)} / {X.shape[1]}")
    print(f"Classes               : {class_names}")
    print(f"CV strategy           : {type(cv).__name__}")
    print(f"Best params (CV)      : {best_params}")
    print(f"CV best accuracy      : {cv_best_acc:.4f}")
    print(f"Test accuracy         : {test_acc:.4f}")
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)
    print("\nSaved artifacts:")
    print(" -", cm_path)
    print(" -", lc_path)
    print(" -", os.path.join(OUT_DIR, "U_A12_classification_report.txt"))
    print(" -", os.path.join(OUT_DIR, "U_A12_summary.json"))
    if _HAVE_JOBLIB:
        print(" -", os.path.join(OUT_DIR, "U_A12_best_model.joblib"))

if __name__ == "__main__":
    main()
