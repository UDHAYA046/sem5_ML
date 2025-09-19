# ------------------------------------------------------------
# Lab 8 – A12 | Use scikit-learn MLPClassifier on project data
# Author : S. Udhaya Sankari | BL.EN.U4CSE23150
# ------------------------------------------------------------

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold,
                                     GridSearchCV, learning_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False

warnings.filterwarnings("ignore")

# -------------- CONFIG --------------
DATA_PATH = "features_lab3_labeled.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", "lab8_output_figures")
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 123

# -------------- HELPERS --------------
def guess_target_column(df: pd.DataFrame) -> str:
    hits = [c for c in df.columns if c.lower() in ("label", "target", "class", "y", "outcome")]
    if hits: return hits[0]
    n = len(df)
    uniq = sorted(((c, df[c].nunique()) for c in df.columns), key=lambda x: x[1])
    for c, u in uniq:
        if 2 <= u <= max(10, int(0.05 * n)):
            return c
    return df.columns[-1]

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        # Make OneHotEncoder compatible with both old/new sklearn:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn < 1.2
        transformers.append(("cat", enc, cat_cols))
    if not transformers:
        raise ValueError("No feature columns detected.")
    return ColumnTransformer(transformers=transformers, remainder="drop")

# -------------- MAIN --------------
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    target_col = guess_target_column(df)
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    preprocessor = build_preprocessor(X)

    base_mlp = MLPClassifier(max_iter=300, random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", preprocessor), ("clf", base_mlp)])

    param_grid = {
        "clf__hidden_layer_sizes": [(16,), (32,), (16, 8)],
        "clf__activation": ["relu", "logistic"],
        "clf__solver": ["adam"],
        "clf__learning_rate_init": [0.001, 0.01],
        "clf__alpha": [0.0001, 0.001],
    }

    # stratify only if all classes have >= 2 samples
    _, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    stratify_y = y if min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y, shuffle=True
    )

    if min_count >= 2:
        n_splits = max(2, int(min(5, min_count)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", refit=True, verbose=0)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    cv_best_acc = grid.best_score_

    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    all_label_ids = np.arange(len(class_names))
    report = classification_report(
        y_test, y_pred, labels=all_label_ids, target_names=class_names, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=all_label_ids)

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(values_format="d")
    plt.title("A12 – Confusion Matrix (MLPClassifier)")
    cm_path = os.path.join(OUT_DIR, "U_A12_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train, cv=cv,
        train_sizes=np.linspace(0.25, 1.0, 5), scoring="accuracy"
    )
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="CV")
    plt.title("A12 – Learning Curve (MLPClassifier)")
    plt.xlabel("Training examples"); plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", linewidth=0.6); plt.legend()
    lc_path = os.path.join(OUT_DIR, "U_A12_learning_curve.png")
    plt.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save reports
    rep_path = os.path.join(OUT_DIR, "U_A12_classification_report.txt")
    with open(rep_path, "w") as f:
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
            "classification_report_txt": os.path.abspath(rep_path),
        },
    }
    summ_path = os.path.join(OUT_DIR, "U_A12_summary.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)

    if HAVE_JOBLIB:
        model_path = os.path.join(OUT_DIR, "U_A12_best_model.joblib")
        joblib.dump(best_model, model_path)
        summary["artifacts"]["best_model_joblib"] = os.path.abspath(model_path)
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Console summary
    print("\n=== A12: MLPClassifier on project dataset ===")
    print(f"Target column         : {target_col}")
    print(f"Samples / Features    : {len(df)} / {X.shape[1]}")
    print(f"Classes               : {class_names}")
    print(f"CV strategy           : {type(cv).__name__}")
    print(f"Best params (CV)      : {best_params}")
    print(f"CV best accuracy      : {cv_best_acc:.4f}")
    print(f"Test accuracy         : {test_acc:.4f}")
    print("\nSaved:")
    print(" -", cm_path)
    print(" -", lc_path)
    print(" -", rep_path)
    print(" -", summ_path)
    if HAVE_JOBLIB:
        print(" -", model_path)

if __name__ == "__main__":
    main()
