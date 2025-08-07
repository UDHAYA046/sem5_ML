import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# ---------- 1. Load Dataset ----------
def load_confidence_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['mfcc1', 'class']].dropna()
    return df[['mfcc1']], df['class']

# ---------- 2. Train-Test Split ----------
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 3. Train Linear Regression ----------
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ---------- 4. Evaluate Model ----------
def evaluate(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    return y_pred, mse, rmse, r2, mape

# ---------- 5. Plot Results ----------
def plot_results(X, y, y_pred, title):
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(X, y_pred, color='red', label='Regression Line')
    plt.xlabel("MFCC1")
    plt.ylabel("Confidence Level")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- 6. Main ----------
if __name__ == "__main__":
    # Update this path if needed
    csv_path = "features_lab3_labeled.csv"

    X, y = load_confidence_dataset(csv_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    # Train evaluation
    y_train_pred, mse_train, rmse_train, r2_train, mape_train = evaluate(model, X_train, y_train)
    plot_results(X_train, y_train, y_train_pred, "Train Set: MFCC1 vs Confidence Level")

    # Test evaluation
    y_test_pred, mse_test, rmse_test, r2_test, mape_test = evaluate(model, X_test, y_test)
    plot_results(X_test, y_test, y_test_pred, "Test Set: MFCC1 vs Confidence Level")

    # Print results outside functions
    print("📊 Train Metrics:")
    print(f"MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}, MAPE: {mape_train:.4f}")
    print("\n📊 Test Metrics:")
    print(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}, MAPE: {mape_test:.4f}")
