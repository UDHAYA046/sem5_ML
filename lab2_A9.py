import pandas as pd

def u_check_missing_values_after_imputation():
    """
    LOADS THE THYROID DATASET (ASSUMED CLEANED) AND
    RETURNS A SERIES SHOWING MISSING VALUES PER COLUMN.
    """

    # Load dataset again (already imputed from A8 or saved as cleaned file)
    file_path = r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx"
    u_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    # Check and return missing value counts
    return u_df.isnull().sum()

if __name__ == "__main__":
    print("\n Missing values after imputation:\n")
    u_missing_summary = u_check_missing_values_after_imputation()
    print(u_missing_summary)
