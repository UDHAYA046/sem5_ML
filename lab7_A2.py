# U_Lab07_A2_min.py
# Lab 07 – A2: Hyperparameter Tuning with RandomizedSearchCV
# Style: all names prefixed with U_, minimal + heavily commented, I/O only in __main__.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Models (same set as your working code) ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- Tools ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ----------------------------- USER SETTINGS -----------------------------
U_DATA_PATH = r"C:\semester 5\machine learning\project\VivaData_Set2_23012\features_lab3_labeled.csv"
U_OUT_DIR   = r"C:\Users\Udhaya\sem5_ML\lab7_output_figures"
U_FEATURES  = ['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']   # selected features
U_TARGET    = 'class'
U_SEED      = 42
U_TEST_SIZE = 0.30
U_N_ITER    = 10
U_CV_FOLDS  = 3
# ------------------------------------------------------------------------


if __name__ == "__main__":
    os.makedirs(U_OUT_DIR, exist_ok=True)

    # -------------------- U1) Load dataset --------------------
    U_df = pd.read_csv(U_DATA_PATH)

    # Feature matrix and target vector (copy to avoid SettingWithCopy warnings)
    U_X = U_df[U_FEATURES].copy()
    U_y = U_df[U_TARGET].copy()

    # Ensure all features are numeric; non-numeric -> NaN, then impute with column means
    for U_c in U_X.columns:
        U_X[U_c] = pd.to_numeric(U_X[U_c], errors="coerce")
    U_X = U_X.fillna(U_X.mean(numeric_only=True))

    # -------------------- U2) Filter ultra-rare classes --------------------
    # Keep classes that appear at least twice so stratified split / CV won't error
    U_counts = U_y.value_counts()
    U_valid_classes = U_counts[U_counts > 1].index
    U_mask = U_y.isin(U_valid_classes)
    U_X = U_X[U_mask]
    U_y = U_y[U_mask]

    # -------------------- U3) Train–test split (stratified) --------------------
    U_Xtr, U_Xte, U_ytr, U_yte = train_test_split(
        U_X, U_y, test_size=U_TEST_SIZE, random_state=U_SEED, stratify=U_y
    )

    # -------------------- U4) Define models + search spaces --------------------
    # Scale-sensitive algorithms wrapped in a Pipeline with StandardScaler
    U_knn = Pipeline([("U_scale", StandardScaler()), ("U_clf", KNeighborsClassifier())])
    U_svm = Pipeline([("U_scale", StandardScaler()), ("U_clf", SVC())])
    U_dt  = DecisionTreeClassifier(random_state=U_SEED)
    U_rf  = RandomForestClassifier(random_state=U_SEED)

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
        "RandomForest": (U_rf, U_rf_params)
    }

    # -------------------- U5) Hyperparameter tuning --------------------
    U_rows = []
    for U_model_name, (U_estimator, U_param_grid) in U_candidates.items():
        # Randomized search over the grid; return_train_score=True enables plotting cv_results_
        U_tuner = RandomizedSearchCV(
            estimator=U_estimator,
            param_distributions=U_param_grid,
            n_iter=U_N_ITER,
            cv=U_CV_FOLDS,
            random_state=U_SEED,
            n_jobs=-1,
            return_train_score=True
        )
        U_tuner.fit(U_Xtr, U_ytr)  # fit on training data only

        # Evaluate best model on the held-out test set
        U_yhat = U_tuner.predict(U_Xte)
        U_acc  = accuracy_score(U_yte, U_yhat)
        U_f1w  = f1_score(U_yte, U_yhat, average="weighted")

        # Save a concise text report per model (best params + classification report)
        U_report_path = os.path.join(U_OUT_DIR, f"Lab07_A2_{U_model_name}_report.txt")
        with open(U_report_path, "w", encoding="utf-8") as U_f:
            U_f.write("Best Params:\n")
            U_f.write(str(U_tuner.best_params_) + "\n\n")
            U_f.write("Classification Report (Test):\n")
            U_f.write(classification_report(U_yte, U_yhat))

        # Keep a summary row for the final comparison table
        U_rows.append({
            "Model": U_model_name,
            "BestParams": U_tuner.best_params_,
            "Best_CV_Score": float(U_tuner.best_score_),
            "Test_Accuracy": float(U_acc),
            "Test_F1_weighted": float(U_f1w)
        })

        # Plot CV trial scores and save (no interactive display)
        U_cv_df = pd.DataFrame(U_tuner.cv_results_)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=np.arange(len(U_cv_df)), y=U_cv_df['mean_test_score'], palette="coolwarm")
        plt.title(f"{U_model_name} - RandomizedSearch CV mean scores")
        plt.xlabel("Trial")
        plt.ylabel("Mean CV Score")
        plt.tight_layout()
        plt.savefig(os.path.join(U_OUT_DIR, f"Lab07_A2_{U_model_name}_CVscores.png"), dpi=200)
        plt.close()

    # -------------------- U6) Save comparison table + final plot --------------------
    U_perf_df = pd.DataFrame(U_rows).sort_values("Test_F1_weighted", ascending=False)
    U_perf_path = os.path.join(U_OUT_DIR, "Lab07_A2_results.csv")
    U_perf_df.to_csv(U_perf_path, index=False)

    # Final comparison figure (Accuracy vs F1 on test)
    plt.figure(figsize=(8, 5))
    U_melted = U_perf_df.melt(id_vars="Model", value_vars=["Test_Accuracy", "Test_F1_weighted"])
    sns.barplot(data=U_melted, x="Model", y="value", hue="variable", palette="Set2")
    plt.title("Tuned Models – Test Set Performance")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(U_OUT_DIR, "Lab07_A2_comparison.png"), dpi=200)
    plt.close()

    # Minimal console summary (allowed here)
    print("Saved:", U_perf_path)
    print(U_perf_df[["Model", "Best_CV_Score", "Test_Accuracy", "Test_F1_weighted"]])
