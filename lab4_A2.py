# ---------------------------- MODULE IMPORTS ----------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# ---------------------------- FUNCTION DEFINITIONS ----------------------------

def U_load_clean_purchase_data(U_file_path):
    """
    Loads and cleans the purchase dataset from Lab 02.
    Returns feature matrix and label vector.
    """
    U_data = pd.read_excel(U_file_path, sheet_name="Purchase data")
    U_required = U_data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]]
    U_cleaned = U_required.dropna()
    U_features = U_cleaned[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
    U_target = U_cleaned["Payment (Rs)"]
    return U_features, U_target

def U_train_linear_regression(U_X, U_y):
    """
    Trains and returns a Linear Regression model.
    """
    U_model = LinearRegression()
    U_model.fit(U_X, U_y)
    return U_model

def U_evaluate_regression(U_model, U_X, U_y_true):
    """
    Calculates and returns MSE, RMSE, MAPE, and R² Score.
    """
    U_y_pred = U_model.predict(U_X)
    U_mse = mean_squared_error(U_y_true, U_y_pred)
    U_rmse = np.sqrt(U_mse)
    U_mape = mean_absolute_percentage_error(U_y_true, U_y_pred)
    U_r2 = r2_score(U_y_true, U_y_pred)

    return {
        "MSE": U_mse,
        "RMSE": U_rmse,
        "MAPE": U_mape,
        "R2 Score": U_r2
    }

# ---------------------------- MAIN PROGRAM ----------------------------

if __name__ == "__main__":
    # Path to Lab 02 Excel file
    U_excel_path = "Lab Session Data.xlsx"

    # Load and prepare data
    U_X_purchase, U_y_purchase = U_load_clean_purchase_data(U_excel_path)

    # Train regression model
    U_model_reg = U_train_linear_regression(U_X_purchase, U_y_purchase)

    # Evaluate model metrics
    U_metrics = U_evaluate_regression(U_model_reg, U_X_purchase, U_y_purchase)

    # Print results
    print(" Lab 02 Price Prediction – Evaluation Metrics")
    for U_key, U_value in U_metrics.items():
        print(f"{U_key}: {U_value}")
