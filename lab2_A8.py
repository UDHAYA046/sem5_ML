import pandas as pd
import numpy as np
from scipy.stats import mode
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def u_perform_data_imputation():
    """
    IMPUTES MISSING VALUES IN THE 'thyroid0387_UCI' DATASET
    USING MEAN, MEDIAN, OR MODE BASED ON ATTRIBUTE TYPE AND OUTLIERS.
    """

    # Load dataset
    file_path = r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx"
    u_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    # Replace '?' with NaN
    u_df.replace('?', np.nan, inplace=True)

    # Store imputation details to display later
    u_imputation_log = []

    # Begin imputation
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
                        u_imputation_log.append(f"{col} - Numeric (Outliers Present) - Filled with MEDIAN ({med})")
                    else:
                        mean = round(u_df[col].mean(), 2)
                        u_df[col] = u_df[col].fillna(mean)
                        u_imputation_log.append(f"{col} - Numeric (No Outliers) - Filled with MEAN ({mean})")
                else:
                    mod = mode(u_df[col].dropna())[0][0]
                    u_df[col] = u_df[col].fillna(mod)
                    u_imputation_log.append(f"{col} - Categorical - Filled with MODE ({mod})")
            except:
                mod = mode(u_df[col].dropna())[0][0]
                u_df[col] = u_df[col].fillna(mod)
                u_imputation_log.append(f"{col} - Categorical (Forced) - Filled with MODE ({mod})")

    return u_df, u_imputation_log

if __name__ == "__main__":
    print("\n Starting Data Imputation...\n")
    u_final_df, u_log = u_perform_data_imputation()
    for entry in u_log:
        print(entry)
    print("\n Imputation complete.")
