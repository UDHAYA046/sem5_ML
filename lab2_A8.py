
import pandas as pd
import numpy as np
from scipy.stats import mode
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
file_path = r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx"
u_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Replace '?' with NaN
u_df.replace('?', np.nan, inplace=True)

# Begin imputation
print("\n Starting Data Imputation...\n")

for col in u_df.columns:
    if u_df[col].isnull().sum() > 0:
        try:
            u_df[col] = pd.to_numeric(u_df[col], errors='coerce')
            if u_df[col].dtype in [np.float64, np.int64]:
                # Check for outliers
                Q1 = u_df[col].quantile(0.25)
                Q3 = u_df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = u_df[(u_df[col] < Q1 - 1.5 * IQR) | (u_df[col] > Q3 + 1.5 * IQR)]

                if len(outliers) > 0:
                    med = u_df[col].median()
                    u_df[col] = u_df[col].fillna(med)
                    print(f"{col} - Numeric (Outliers Present) - Filled with MEDIAN ({med})")
                else:
                    mean = round(u_df[col].mean(), 2)
                    u_df[col] = u_df[col].fillna(mean)
                    print(f"{col} - Numeric (No Outliers) - Filled with MEAN ({mean})")
            else:
                mod = mode(u_df[col].dropna())[0][0]
                u_df[col] = u_df[col].fillna(mod)
                print(f"{col} - Categorical - Filled with MODE ({mod})")
        except:
            mod = mode(u_df[col].dropna())[0][0]
            u_df[col] = u_df[col].fillna(mod)
            print(f"{col} - Categorical (Forced) - Filled with MODE ({mod})")

print("\n Imputation complete.")
