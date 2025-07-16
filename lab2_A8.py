import pandas as pd
import numpy as np
from scipy.stats import zscore

# Load dataset from your path
u_df = pd.read_excel(r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx", sheet_name=0)

# Replace '?' with NaN
u_df.replace('?', np.nan, inplace=True)

# Convert columns to numeric where possible
for col in u_df.columns:
    try:
        u_df[col] = pd.to_numeric(u_df[col])
    except:
        continue

print("\nMISSING VALUE IMPUTATION STRATEGY:")
print("===================================")

# Impute missing values using mean, median, or mode based on type and outliers
for col in u_df.columns:
    if u_df[col].isnull().sum() > 0:
        if u_df[col].dtype == 'object':
            mode_val = u_df[col].mode()[0]
            u_df[col].fillna(mode_val, inplace=True)
            print(f"{col} - Categorical - Filled with MODE ({mode_val})")
        elif np.issubdtype(u_df[col].dtype, np.number):
            non_null = u_df[col].dropna()
            z_scores = np.abs(zscore(non_null))
            if any(z_scores > 3):
                median_val = non_null.median()
                u_df[col].fillna(median_val, inplace=True)
                print(f"{col} - Numeric with Outliers - Filled with MEDIAN ({median_val})")
            else:
                mean_val = non_null.mean()
                u_df[col].fillna(mean_val, inplace=True)
                print(f"{col} - Numeric (No Outliers) - Filled with MEAN ({mean_val})")

print("\n✅ Missing values handled successfully.")
