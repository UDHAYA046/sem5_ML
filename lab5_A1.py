# Lab05 A1 – Linear Regression on single feature (mfcc1)
# Author: Udhaya 

import pandas as U_pd
import matplotlib.pyplot as U_plt
from sklearn.linear_model import LinearRegression as U_LinearRegression
from sklearn.model_selection import train_test_split as U_train_test_split

# --- Load and prepare dataset ---
def U_load_mfcc1_dataset(U_path):
    U_data = U_pd.read_csv(U_path)
    return U_data[['mfcc1']], U_data['class']

# --- Split dataset ---
def U_split_train_test(U_X, U_y):
    return U_train_test_split(U_X, U_y, test_size=0.2, random_state=42)

# --- Train linear regression model ---
def U_train_single_feature_model(U_X_train, U_y_train):
    U_model = U_LinearRegression()
    U_model.fit(U_X_train, U_y_train)
    return U_model

# --- Plot prediction line ---
def U_plot_regression_line(U_X, U_y, U_y_pred, U_title):
    U_plt.figure(figsize=(8, 5))
    U_plt.scatter(U_X, U_y, color='blue', label='Actual')
    U_plt.plot(U_X, U_y_pred, color='red', label='Regression Line')
    U_plt.xlabel('MFCC1')
    U_plt.ylabel('Confidence Level')
    U_plt.title(U_title)
    U_plt.grid(True)
    U_plt.legend()
    U_plt.tight_layout()
    U_plt.show()

# --- Main execution for A1 ---
if __name__ == "__main__":
    U_csv_path = "features_lab3_labeled.csv"
    U_X, U_y = U_load_mfcc1_dataset(U_csv_path)
    U_X_train, U_X_test, U_y_train, U_y_test = U_split_train_test(U_X, U_y)

    U_reg_model = U_train_single_feature_model(U_X_train, U_y_train)

    # Predictions
    U_y_train_pred = U_reg_model.predict(U_X_train)
    U_y_test_pred = U_reg_model.predict(U_X_test)

    # Plot results
    U_plot_regression_line(U_X_train, U_y_train, U_y_train_pred, "Train Set: MFCC1 vs Confidence")
    U_plot_regression_line(U_X_test, U_y_test, U_y_test_pred, "Test Set: MFCC1 vs Confidence")
