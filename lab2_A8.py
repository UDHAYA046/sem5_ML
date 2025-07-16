import pandas as pd
import numpy as np
from scipy.stats import zscore

# Load your Excel file (update the correct path)
file_path = r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx"
u_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Replace '?' with NaN
u_df.replace('?', np.nan, inplace=True)

# Convert numeric columns to appropriate types (for mean/median calculations)
for col in u_df.columns:
    u_df[col] = pd.to_numeric(u_df[col], errors='ignore')

# Imputation logic
for col in u_df.columns:
    if u_df[col].isnull().sum() > 0:  # Only if missing values exist
        if u_df[col].dtype == 'object':
            mode_val = u_df[col].mode()[0]
            u_df[col] = u_df[col].fillna(mode_val)
            print(f"{col} - Categorical - Filled with MODE ({mode_val})")
        elif np.issubdtype(u_df[col].dtype, np.number):
            non_null = u_df[col].dropna()
            z_scores = np.abs(zscore(non_null))
            if any(z_scores > 3):
                median_val = non_null.median()
                u_df[col] = u_df[col].fillna(median_val)
                print(f"{col} - Numeric (Outliers Present) - Filled with MEDIAN ({median_val})")
            else:
                mean_val = non_null.mean()
                u_df[col] = u_df[col].fillna(mean_val)
                print(f"{col} - Numeric (No Outliers) - Filled with MEAN ({mean_val})")

print("\n✅ Missing values handled successfully.")
