import pandas as pd
import numpy as np
from scipy.stats import mode

# Replace with your actual DataFrame
# u_df = pd.read_excel("your_excel_file.xlsx", sheet_name="thyroid0387_UCI")

# Replace '?' with NaN
u_df.replace('?', np.nan, inplace=True)

# Convert all columns to appropriate types (if possible)
for col in u_df.columns:
    u_df[col] = pd.to_numeric(u_df[col], errors='ignore')

# Function to detect outliers using IQR method
def has_outliers(series):
    if series.dtype.kind in 'biufc':  # only numeric
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return series[(series < lower_bound) | (series > upper_bound)].any()
    return False

# Impute missing values
for col in u_df.columns:
    if u_df[col].isnull().any():
        if u_df[col].dtype.kind in 'biufc':  # Numeric
            if has_outliers(u_df[col].dropna()):
                value = u_df[col].median()
                strategy = "Median"
            else:
                value = u_df[col].mean()
                strategy = "Mean"
        else:  # Categorical
            value = u_df[col].mode().iloc[0]
            strategy = "Mode"
        print(f"Filling missing values in '{col}' using {strategy}: {value}")
        u_df[col].fillna(value, inplace=True)

# Display updated DataFrame (optional)
print("\nUpdated DataFrame after Imputation:")
print(u_df.head())
