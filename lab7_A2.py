# U_Lab07_A2.py
# Lab 07 – A2: Hyperparameter Tuning with RandomizedSearchCV
# Style rules: U_ prefix for all names; functions defined before __main__; no prints inside functions.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ------------------------- CONFIG (edit if needed) -------------------------
U_DATA_PATH = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
U_OUT_DIR   = r"C:\Users\Udhaya\sem5_ML\lab7_output_figures"
U_FEATURES  = ['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']
U_TARGET    = 'class'
U_SEED      = 42
U_TEST_SIZE = 0.30
U_N_ITER    = 10
U_CV_FOLDS  = 3
# --------------------------------------------------------------------------


# ------------------------------ FUNCTIONS ---------------------------------
def U_load_dataframe(csv_path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def U_select_and_clean_features(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Select given features and target, coerce features to numeric,
    and impute missing values with column means.
    Returns X (DataFrame) and y (Series).
    """
    # keep only columns that exist
    U_present = [c for c in feature_cols if c in df.columns]
    U_X = df[U_present].copy()
    U_y = df[target_col].copy()

    # numeric coercion + mean impute (handles accidental strings)
    for U_c in U_X.columns:
        U_X[U_c] = pd.to_numeric(U_X[U_c], errors="coerce")
    U_X = U_X.fillna(U_X.mean(numeric_only=True))
    return U_X, U_y


def U_filter_rare_classes(X: pd.DataFrame, y: pd.Series, min_count: int = 2):
    """
    Keep only classes with at least 'min_count' samples (needed for stratify/CV).
    Returns filtered X, y.
    """
    U_counts = y.value_counts()
    U_valid = U_counts[U_counts >= min_count].index
    U_mask = y.isin(U_valid)
    return X[U_mask], y[U_mask]


def U_split_data(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """Stratified train/test split (safe because rare classes already filtered)."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def U_build_candidates(seed: int):
    """
    Build estimators and their RandomizedSearch spaces.
    KNN + SVM are wrapped with StandardScaler via Pipeline.
    """
    U_knn = Pipeline([("U_scale", StandardScaler()), ("U_clf", KNeighborsClassifier())])
    U_svm = Pipeline([("U_scale", StandardScaler()), ("U_clf", SVC())])
    U_dt  = DecisionTreeClassifier(random_state=seed)
    U_rf  = RandomForestClassifier(random_state=seed)

    U_knn_params = {
        'U_clf__n_neighbors': np.arange(1, 15),
        'U_clf__weights': ['uniform', 'distance'],
        'U_clf__metric': ['euclidean', 'manhattan', 'minkowski']
    }
    U_svm_params = {
        'U_clf__C': [0.1, 1, 10, 50, 100],
        'U_clf__gamma': ['scale', 'auto'],
        'U_clf__kernel': ['linear', 'rbf', 'poly']
    }
    U_dt_params = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy', 'log_loss']
    }
    U_rf_params = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    U_candidates = {
        "KNN": (U_knn, U_knn_params),
        "SVM": (U_svm, U_svm_params),
        "DecisionTree": (U_dt, U_dt_params),
        "RandomForest": (U_rf, U_rf_params),
    }
    return U_candidates


def U_tune_model(estimator, param_grid, Xtr, ytr, n_iter: int, cv_folds: int, seed: int):
    """
    Run RandomizedSearchCV and return the fitted tuner.
    (No printing here; caller inspects attributes.)
    """
    U_tuner = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_folds,
        random_state=seed,
        n_jobs=-1,
        return_train_score=True
    )
    U_tuner.fit(Xtr, ytr)
    return U_tuner


def U_eval_on_test(tuner: RandomizedSearchCV, Xte: pd.DataFrame, yte: pd.Series):
    """
    Evaluate tuned model on the test set and return a summary dict
    plus the classification report text.
    """
    U_yhat = tuner.predict(Xte)
    U_acc = accuracy_score(yte, U_yhat)
    U_f1w = f1_score(yte, U_yhat, average="weighted")
    U_report = classification_report(yte, U_yhat)
    return {
        "BestParams": tuner.best_params_,
        "Best_CV_Score": float(tuner.best_score_),
        "Test_Accuracy": float(U_acc),
        "Test_F1_weighted": float(U_f1w)
    }, U_report


def U_save_text_report(out_dir: str, model_name: str, best_params: dict, report_text: str):
    """Save best params + classification report to a text file."""
    U_path = os.path.join(out_dir, f"Lab07_A2_{model_name}_report.txt")
    with open(U_path, "w", encoding="utf-8") as U_f:
        U_f.write("Best Params:\n")
        U_f.write(str(best_params) + "\n\n")
        U_f.write("Classification Report (Test):\n")
        U_f.write(report_text)
    return U_path


def U_save_cv_plot(out_dir: str, model_name: str, tuner: RandomizedSearchCV):
    """Save a bar plot of mean CV scores across RandomizedSearch trials."""
    U_cv = pd.DataFrame(tuner.cv_results_)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=np.arange(len(U_cv)), y=U_cv['mean_test_score'], palette="coolwarm")
    plt.title(f"{model_name} - RandomizedSearch CV mean scores")
    plt.xlabel("Trial")
    plt.ylabel("Mean CV Score")
    plt.tight_layout()
    U_path = os.path.join(out_dir, f"Lab07_A2_{model_name}_CVscores.png")
    plt.savefig(U_path, dpi=200)
    plt.close()
    return U_path


def U_save_summary_table(out_dir: str, rows: list):
    """Save the final comparison table as CSV and return its path and DataFrame."""
    U_df = pd.DataFrame(rows).sort_values("Test_F1_weighted", ascending=False)
    U_path = os.path.join(out_dir, "Lab07_A2_results.csv")
    U_df.to_csv(U_path, index=False)
    return U_path, U_df


def U_save_final_bar(out_dir: str, df: pd.DataFrame):
    """Save a grouped bar chart (Test Accuracy vs Test F1-weighted)."""
    plt.figure(figsize=(8, 5))
    U_melt = df.melt(id_vars="Model", value_vars=["Test_Accuracy", "Test_F1_weighted"])
    sns.barplot(data=U_melt, x="Model", y="value", hue="variable", palette="Set2")
    plt.title("Tuned Models – Test Set Performance")
    plt.ylabel("Score")
    plt.tight_layout()
    U_path = os.path.join(out_dir, "Lab07_A2_comparison.png")
    plt.savefig(U_path, dpi=200)
    plt.close()
    return U_path
# -------------------------------------------------------------------------


# --------------------------------- MAIN ----------------------------------
if __name__ == "__main__":
    # ensure paths exist / are valid
    os.makedirs(U_OUT_DIR, exist_ok=True)
    assert os.path.exists(U_DATA_PATH), f"CSV not found at: {U_DATA_PATH}"

    # 1) Load and prepare data
    U_df = U_load_dataframe(U_DATA_PATH)
    U_X, U_y = U_select_and_clean_features(U_df, U_FEATURES, U_TARGET)
    U_X, U_y = U_filter_rare_classes(U_X, U_y, min_count=2)

    # 2) Split data (stratified)
    U_Xtr, U_Xte, U_ytr, U_yte = U_split_data(U_X, U_y, U_TEST_SIZE, U_SEED)

    # 3) Build models + search spaces
    U_candidates = U_build_candidates(U_SEED)

    # 4) Tune each model, evaluate, and save artifacts
    U_rows = []
    for U_name, (U_est, U_space) in U_candidates.items():
        U_tuner = U_tune_model(U_est, U_space, U_Xtr, U_ytr, U_N_ITER, U_CV_FOLDS, U_SEED)
        U_summary, U_report_text = U_eval_on_test(U_tuner, U_Xte, U_yte)
        U_save_text_report(U_OUT_DIR, U_name, U_summary["BestParams"], U_report_text)
        U_save_cv_plot(U_OUT_DIR, U_name, U_tuner)
        U_rows.append({"Model": U_name, **U_summary})

    # 5) Save final table and bar chart
    U_csv_path, U_perf_df = U_save_summary_table(U_OUT_DIR, U_rows)
    U_fig_path = U_save_final_bar(U_OUT_DIR, U_perf_df)

    # 6) Console summary (allowed here)
    print("Saved table ->", U_csv_path)
    print("Saved figure ->", U_fig_path)
    print(U_perf_df[["Model", "Best_CV_Score", "Test_Accuracy", "Test_F1_weighted"]])
