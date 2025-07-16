
import pandas as pd

# LOAD THE DATASET FROM THE EXCEL FILE AND WORKSHEET
u_file_path = "Lab Session Data.xlsx"
u_df_full = pd.read_excel(u_file_path, sheet_name="purchase_data")

# SELECT NUMERIC COLUMNS ONLY FOR MATRIX OPERATIONS
u_df_numeric = u_df_full.select_dtypes(include=['float64', 'int64'])

# EXTRACT TWO SQUARE MATRICES OF SHAPE (4 x 4) FROM DIFFERENT REGIONS
u_square_matrix_A = u_df_numeric.iloc[0:4, 0:4]  # TOP LEFT 4x4 BLOCK
u_square_matrix_B = u_df_numeric.iloc[4:8, 4:8]  # ANOTHER 4x4 BLOCK

# COMPUTE COVARIANCE MATRICES
u_cov_matrix_A = u_square_matrix_A.cov()
u_cov_matrix_B = u_square_matrix_B.cov()
u_cov_matrix_full = u_df_numeric.cov()

#  COMPUTE CORRELATION MATRICES
u_corr_matrix_A = u_square_matrix_A.corr()
u_corr_matrix_B = u_square_matrix_B.corr()
u_corr_matrix_full = u_df_numeric.corr()

#  DISPLAY RESULTS FOR COMPARISON
print(" COVARIANCE MATRIX – Square Matrix A (Top Left):\n", u_cov_matrix_A, "\n")
print(" COVARIANCE MATRIX – Square Matrix B:\n", u_cov_matrix_B, "\n")
print(" COVARIANCE MATRIX – FULL DATASET:\n", u_cov_matrix_full, "\n")

print(" CORRELATION MATRIX – Square Matrix A (Top Left):\n", u_corr_matrix_A, "\n")
print(" CORRELATION MATRIX – Square Matrix B:\n", u_corr_matrix_B, "\n")
print(" CORRELATION MATRIX – FULL DATASET:\n", u_corr_matrix_full, "\n")
