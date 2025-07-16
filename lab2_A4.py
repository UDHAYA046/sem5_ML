import pandas as pd
import numpy as np

def u_explore_thyroid_dataset():
    """
    LOADS THE 'thyroid0387_UCI' DATASET AND PERFORMS EXPLORATORY DATA ANALYSIS
    AS REQUIRED IN TASK A4 INCLUDING TYPE CHECKING, MISSING VALUES, RANGE,
    OUTLIERS, AND BASIC STATS FOR NUMERIC ATTRIBUTES.
    """

    # LOAD DATA FROM EXCEL WORKSHEET
    u_df = pd.read_excel(
        r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI"
    )

    # STRIP COLUMN NAMES TO REMOVE SPACES
    u_df.columns = u_df.columns.str.strip()

    # REPLACE '?' WITH np.nan TO HANDLE MISSING VALUES
    pd.set_option('future.no_silent_downcasting', True)
    u_df = u_df.replace('?', np.nan)
    


    # DISPLAY ATTRIBUTE TYPE SUGGESTIONS (CLEAN FORMAT WITH DASHES)

    print("\nATTRIBUTE TYPE SUGGESTIONS:\n")

    # CATEGORIZE COLUMNS
    u_summary = {
        "Numeric": ['Record ID', 'age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG'],
        "Binary": [col for col in u_df.columns if set(u_df[col].dropna().unique()) <= {'f', 't'}],
        "Nominal": ['sex', 'referral source'],
        "Target": ['Condition']
    }

    # CLEAN FORMATTED PRINTING
    for u_type, u_cols in u_summary.items():
        for u_col in u_cols:
            if u_type == "Binary":
                print(f"{u_col:<25} - Binary (f=0, t=1)")
            elif u_type == "Nominal":
                print(f"{u_col:<25} - Nominal (One-Hot)")
            elif u_type == "Target":
                print(f"{u_col:<25} - Target (Nominal)")
            else:
                print(f"{u_col:<25} - Numeric")


    # CHECK FOR MISSING VALUES
    print("\n MISSING VALUES PER COLUMN:")
    for u_col in u_df.columns:
        u_missing = u_df[u_col].isna().sum()
        if u_missing > 0:
            print(f" {u_col}: {u_missing} missing")

    # IDENTIFY NUMERIC COLUMNS
    u_numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    u_existing_numeric_cols = [col for col in u_numeric_cols if col in u_df.columns]

    # CONVERT NUMERIC COLUMNS TO FLOAT
    for u_col in u_existing_numeric_cols:
        u_df[u_col] = pd.to_numeric(u_df[u_col], errors='coerce')

    # RANGE, MEAN, VARIANCE AND OUTLIERS
    print("\n NUMERIC RANGE, MEAN, STD, OUTLIERS:")
    for u_col in u_existing_numeric_cols:
        u_data = u_df[u_col].dropna()
        if len(u_data) == 0:
            continue

        u_min = u_data.min()
        u_max = u_data.max()
        u_mean = round(u_data.mean(), 2)
        u_std = round(u_data.std(), 2)

        u_upper_limit = u_mean + 3 * u_std
        u_lower_limit = u_mean - 3 * u_std
        u_outliers = u_data[(u_data < u_lower_limit) | (u_data > u_upper_limit)]

        print(f" {u_col}:")
        print(f"     Min: {u_min}, Max: {u_max}, Mean: {u_mean}, Std Dev: {u_std}")
        print(f"     Outliers detected: {len(u_outliers)}")

# RUN THE FUNCTION
u_explore_thyroid_dataset()
