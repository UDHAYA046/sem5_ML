# Lab05 A3 – Multi-feature Linear Regression on confidence prediction
# Author: Udhaya 

import pandas as U_pd
import numpy as U_np
from sklearn.linear_model import LinearRegression as U_LinearRegression
from sklearn.model_selection import train_test_split as U_train_test_split
from sklearn.metrics import mean_squared_error as U_mse
from sklearn.metrics import mean_absolute_percentage_error as U_mape
from sklearn.metrics import r2_score as U_r2

# --------------------- Function 1: Load all features ---------------------
def U_load_multifeature_dataset(U_file_path):
    """
    Loads dataset with all useful numerical features.
    """
    U_df = U_pd.read_csv(U_file_path)
    U_features = U_df[['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']]
    U_target = U_df['class']
    return U_features, U_target

# --------------------- Function 2: Split data ---------------------
def U_split_multifeature_data(U_X, U_y):
    """
    Splits data into 80-20 train-test sets.
    """
    return U_train_test_split(U_X, U_y, test_size=0.2, random_state=42)

# --------------------- Function 3: Train model ---------------------
def U_train_multi_regression(U_X_train, U_y_train):
    """
    Trains linear regression model using multiple features.
    """
    U_model = U_LinearRegression()
    U_model.fit(U_X_train, U_y_train)
    return U_model

# --------------------- Function 4: Evaluate metrics ---------------------
def U_calculate_regression_metrics(U_y_true, U_y_pred):
    """
    Returns MSE, RMSE, R², and MAPE.
    """
    U_mse_val = U_mse(U_y_true, U_y_pred)
    U_rmse_val = U_np.sqrt(U_mse_val)
    U_r2_val = U_r2(U_y_true, U_y_pred)
    U_mape_val = U_mape(U_y_true, U_y_pred)
    return U_mse_val, U_rmse_val, U_r2_val, U_mape_val

# --------------------- Main execution block ---------------------
if __name__ == "__main__":
    U_data_path = "features_lab3_labeled.csv"

    # Load dataset
    U_X, U_y = U_load_multifeature_dataset(U_data_path)
    U_X_train, U_X_test, U_y_train, U_y_test = U_split_multifeature_data(U_X, U_y)

    # Train model
    U_multi_model = U_train_multi_regression(U_X_train, U_y_train)

    # Predict
    U_y_train_pred = U_multi_model.predict(U_X_train)
    U_y_test_pred = U_multi_model.predict(U_X_test)

    # Evaluate train
    U_mse_tr, U_rmse_tr, U_r2_tr, U_mape_tr = U_calculate_regression_metrics(U_y_train, U_y_train_pred)

    # Evaluate test
    U_mse_ts, U_rmse_ts, U_r2_ts, U_mape_ts = U_calculate_regression_metrics(U_y_test, U_y_test_pred)

    # Print results
    print("\n U_Train Metrics (All Features):")
    print(f"MSE: {U_mse_tr:.4f} | RMSE: {U_rmse_tr:.4f} | R²: {U_r2_tr:.4f} | MAPE: {U_mape_tr:.4f}")

    print("\n U_Test Metrics (All Features):")
    print(f"MSE: {U_mse_ts:.4f} | RMSE: {U_rmse_ts:.4f} | R²: {U_r2_ts:.4f} | MAPE: {U_mape_ts:.4f}")
