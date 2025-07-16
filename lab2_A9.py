
import pandas as pd

# Load dataset again (already imputed from A8 or saved as cleaned file)
file_path = r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx"
u_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Optional: if you're reading from the cleaned version, skip replacing '?'

# Final Missing Value Check
print("\n Missing values after imputation:\n")
print(u_df.isnull().sum())
