# Lab05 A2 – Metric Evaluation for A1 Model (mfcc1 regression)
# Author: Udhaya 

import pandas as U_pd
import numpy as U_np
from sklearn.linear_model import LinearRegression as U_LinearRegression
from sklearn.model_selection import train_test_split as U_train_test_split
from sklearn.metrics import mean_squared_error as U_mse
from sklearn.metrics import mean_absolute_percentage_error as U_mape
from sklearn.metrics import r2_score as U_r2

# --- Load and prepare data ---
def U_load_mfcc1_dataset(U_path):
    U_data = U_pd.read_csv(U_path)
    return U_data[['mfcc1']], U_data['class']

# --- Train/Test Split ---
def U_split_data(U_X, U_y):
    return U_train_test_split(U_X, U_y, test_size=0.2, random_state=42)

# --- Train Model ---
def U_train_model(U_X_train, U_y_train):
    U_model = U_LinearRegression()
    U_model.fit(U_X_train, U_y_train)
    return U_model

# --- Evaluate metrics ---
def U_compute_metrics(U_actual, U_predicted):
    U_mse_val = U_mse(U_actual, U_predicted)
    U_rmse_val = U_np.sqrt(U_mse_val)
    U_r2_val = U_r2(U_actual, U_predicted)
    U_mape_val = U_mape(U_actual, U_predicted)
    return U_mse_val, U_rmse_val, U_r2_val, U_mape_val

# --- Main for A2 only (metrics only) ---
if __name__ == "__main__":
    U_csv_path = "features_lab3_labeled.csv"
    U_X, U_y = U_load_mfcc1_dataset(U_csv_path)
    U_X_train, U_X_test, U_y_train, U_y_test = U_split_data(U_X, U_y)

    U_model = U_train_model(U_X_train, U_y_train)

    # Predictions
    U_y_train_pred = U_model.predict(U_X_train)
    U_y_test_pred = U_model.predict(U_X_test)

    # Metrics
    U_mse_train, U_rmse_train, U_r2_train, U_mape_train = U_compute_metrics(U_y_train, U_y_train_pred)
    U_mse_test, U_rmse_test, U_r2_test, U_mape_test = U_compute_metrics(U_y_test, U_y_test_pred)

    # Final results
    print("\n Train Metrics:")
    print(f"MSE: {U_mse_train:.4f}, RMSE: {U_rmse_train:.4f}, R²: {U_r2_train:.4f}, MAPE: {U_mape_train:.4f}")

    print("\n Test Metrics:")
    print(f"MSE: {U_mse_test:.4f}, RMSE: {U_rmse_test:.4f}, R²: {U_r2_test:.4f}, MAPE: {U_mape_test:.4f}")
