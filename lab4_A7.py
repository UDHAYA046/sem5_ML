import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

#  Load and filter dataset (only class 1 and 2)
def fetch_filtered_dataset(csv_file):
    df = pd.read_csv(csv_file)
    filtered_df = df[df['confidence_level'].isin([1, 2])]
    features = filtered_df[['mfcc1', 'pitch_std']].values
    labels = filtered_df['confidence_level'].values
    return features, labels

#  Train-test split wrapper
def partition_data(features, labels, test_ratio=0.3, seed=42):
    return train_test_split(features, labels, test_size=test_ratio, random_state=seed)

#  GridSearchCV wrapper to tune k
def tune_k_via_grid(X_train, y_train, folds=5):
    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(1, 21))}
    grid_search = GridSearchCV(model, param_grid, cv=folds, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_['n_neighbors'], grid_search.cv_results_

#  RandomizedSearchCV wrapper to tune k
def tune_k_via_random(X_train, y_train, k_range=50, folds=5, n_iter=10):
    model = KNeighborsClassifier()
    param_dist = {'n_neighbors': list(range(1, k_range + 1))}
    rand_search = RandomizedSearchCV(model, param_distributions=param_dist, 
                                     n_iter=n_iter, cv=folds, scoring='accuracy', random_state=1)
    rand_search.fit(X_train, y_train)
    return rand_search.best_params_['n_neighbors'], rand_search.cv_results_

#  Plot cross-validation scores from GridSearchCV
def draw_cv_accuracy_plot(cv_results, output_dir, label):
    k_values = cv_results['param_n_neighbors'].data
    scores = cv_results['mean_test_score']
    
    plt.figure()
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('k value')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title(f'Hyperparameter Tuning - {label}')
    plt.grid(True)
    filename = f"A6_A7_cv_accuracy_{label.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

#  MAIN BLOCK
if __name__ == "__main__":
    #  File paths
    csv_file_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    results_folder = r"C:\Users\Udhaya\sem5_ML\lab4_output_figures"
    os.makedirs(results_folder, exist_ok=True)

    #  Load and split
    X, y = fetch_filtered_dataset(csv_file_path)
    X_train, X_test, y_train, y_test = partition_data(X, y)

    #  Grid Search Tuning
    best_k_grid, grid_results = tune_k_via_grid(X_train, y_train)
    print("Best k from GridSearchCV:", best_k_grid)
    draw_cv_accuracy_plot(grid_results, results_folder, label="GridSearchCV")

    #  Random Search Tuning
    best_k_random, random_results = tune_k_via_random(X_train, y_train)
    print("Best k from RandomizedSearchCV:", best_k_random)
    draw_cv_accuracy_plot(random_results, results_folder, label="RandomizedSearchCV")
