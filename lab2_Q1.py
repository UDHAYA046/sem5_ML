import pandas as pd

def u_compute_matrices_from_purchase_data(file_path):
    """
    COMPUTES COVARIANCE AND CORRELATION MATRICES FOR SELECTED
    REGIONS OF THE 'PURCHASE DATA' SHEET.
    
    RETURNS:
    - Covariance matrices for Matrix A, B, and full dataset
    - Correlation matrices for Matrix A, B, and full dataset
    """

    # Load the dataset from the Excel file
    u_df_full = pd.read_excel(file_path, sheet_name="Purchase data")

    # Select numeric columns only for matrix operations
    u_df_numeric = u_df_full.select_dtypes(include=['float64', 'int64'])

    # Extract two square matrices of shape (4 x 4) from different regions
    u_square_matrix_A = u_df_numeric.iloc[0:4, 0:4]  # Top Left 4x4 Block
    u_square_matrix_B = u_df_numeric.iloc[4:8, 4:8]  # Another 4x4 Block

    # Compute covariance matrices
    u_cov_matrix_A = u_square_matrix_A.cov()
    u_cov_matrix_B = u_square_matrix_B.cov()
    u_cov_matrix_full = u_df_numeric.cov()

    # Compute correlation matrices
    u_corr_matrix_A = u_square_matrix_A.corr()
    u_corr_matrix_B = u_square_matrix_B.corr()
    u_corr_matrix_full = u_df_numeric.corr()

    return u_cov_matrix_A, u_cov_matrix_B, u_cov_matrix_full, u_corr_matrix_A, u_corr_matrix_B, u_corr_matrix_full


if __name__ == "__main__":
    u_file_path = "Lab Session Data.xlsx"

    # Call the function
    (u_cov_matrix_A, u_cov_matrix_B, u_cov_matrix_full,
     u_corr_matrix_A, u_corr_matrix_B, u_corr_matrix_full) = u_compute_matrices_from_purchase_data(u_file_path)

    # Display results
    print("COVARIANCE MATRIX – Square Matrix A (Top Left):\n", u_cov_matrix_A, "\n")
    print("COVARIANCE MATRIX – Square Matrix B:\n", u_cov_matrix_B, "\n")
    print("COVARIANCE MATRIX – Full Dataset:\n", u_cov_matrix_full, "\n")

    print("CORRELATION MATRIX – Square Matrix A (Top Left):\n", u_corr_matrix_A, "\n")
    print("CORRELATION MATRIX – Square Matrix B:\n", u_corr_matrix_B, "\n")
    print("CORRELATION MATRIX – Full Dataset:\n", u_corr_matrix_full, "\n")
