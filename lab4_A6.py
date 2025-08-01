import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Function to load and filter dataset
def load_filtered_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['confidence_level'].isin([1, 2])]  # Only retain class 1 and 2
    X = df[['mfcc1', 'pitch_std']].values
    y = df['confidence_level'].values
    return X, y

# Function to split dataset
def get_train_test_split(X, y, test_ratio=0.3, seed=42):
    return train_test_split(X, y, test_size=test_ratio, random_state=seed)

# Function to plot training data
def plot_training(X_train, y_train, save_path):
    plt.figure()
    for label in np.unique(y_train):
        subset = X_train[y_train == label]
        plt.scatter(subset[:, 0], subset[:, 1], label=f'Class {label}', edgecolor='k')
    plt.xlabel('mfcc1')
    plt.ylabel('pitch_std')
    plt.title('A3 Training Data')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'A6_A3_training_data.png'))
    plt.close()

# Function to plot test data
def plot_test(X_test, y_test, save_path):
    plt.figure()
    for label in np.unique(y_test):
        subset = X_test[y_test == label]
        plt.scatter(subset[:, 0], subset[:, 1], label=f'Class {label}', edgecolor='k')
    plt.xlabel('mfcc1')
    plt.ylabel('pitch_std')
    plt.title('A4 Test Data')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'A6_A4_test_data.png'))
    plt.close()

# Function to plot decision boundaries for multiple k
def visualize_knn_decision_regions(X_train, y_train, k_list, save_path):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    for k in k_list:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
        for label in np.unique(y_train):
            subset = X_train[y_train == label]
            plt.scatter(subset[:, 0], subset[:, 1], label=f'Class {label}', edgecolor='k')
        plt.xlabel('mfcc1')
        plt.ylabel('pitch_std')
        plt.title(f'A5: Decision Region using kNN (k={k})')
        plt.legend()
        filename = f'A6_A5_decision_boundary_k{k}.png'
        plt.savefig(os.path.join(save_path, filename))
        plt.close()

# Main driver block
if __name__ == "__main__":
    csv_path ="C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"

    output_path ="C:\Users\Udhaya\sem5_ML\lab4_output_figures"
    os.makedirs(output_path, exist_ok=True)

    X, y = load_filtered_data(csv_path)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    plot_training(X_train, y_train, output_path)
    plot_test(X_test, y_test, output_path)
    visualize_knn_decision_regions(X_train, y_train, k_list=[2, 4, 5, 6], save_path=output_path)
