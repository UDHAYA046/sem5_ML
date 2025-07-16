import pandas as pd
import numpy as np
from scipy.stats import zscore

# Load the Excel data
file_path = r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx"
u_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Replace '?' with NaN
u_df = u_df.replace('?', np.nan)

# Convert all columns to numeric where possible
for col in u_df.columns:
    try:
        u_df[col] = pd.to_numeric(u_df[col])
    except:
        pass  # Ignore columns that can't be converted

# Impute missing values
for col in u_df.columns:
    if u_df[col].isnull().sum() > 0:
        if u_df[col].dtype == 'object':
            mode_val = u_df[col].mode()[0]
            u_df[col] = u_df[col].fillna(mode_val)
            print(f"{col} - Categorical - Filled with MODE ({mode_val})")
        elif np.issubdtype(u_df[col].dtype, np.number):
            non_null = u_df[col].dropna()
            if len(non_null) > 0:
                z_scores = np.abs(zscore(non_null))
                if any(z_scores > 3):
                    median_val = non_null.median()
                    u_df[col] = u_df[col].fillna(median_val)
                    print(f"{col} - Numeric (Outliers Present) - Filled with MEDIAN ({median_val})")
                else:
                    mean_val = non_null.mean()
                    u_df[col] = u_df[col].fillna(mean_val)
                    print(f"{col} - Numeric (No Outliers) - Filled with MEAN ({mean_val})")

print("\n Missing values handled successfully.")
