import pandas as pd
import numpy as np

def u_compute_product_prices_from_purchases():
    """
    LOADS THE 'PURCHASE DATA' SHEET, CONSTRUCTS MATRICES A AND C,
    AND ANSWERS:
    - DIMENSIONALITY OF VECTOR SPACE
    - NUMBER OF VECTORS
    - RANK OF MATRIX A
    - COST PER PRODUCT USING PSEUDO-INVERSE
    """

    # LOAD THE EXCEL FILE i.e. THE SOURCE OF RAW PURCHASE DATA
    u_excel_data = pd.read_excel(
        r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx", 
        sheet_name="Purchase data"
    )

    # EXTRACT MATRIX A i.e. QUANTITIES OF CANDIES, MANGOES, AND MILK
    u_quantity_matrix_A = u_excel_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()

    # EXTRACT VECTOR C i.e. PAYMENT VALUES FOR EACH CUSTOMER
    u_payment_vector_C = u_excel_data[['Payment (Rs)']].to_numpy()

    # DETERMINE DIMENSIONALITY i.e. NUMBER OF FEATURES (PRODUCT TYPES)
    u_dimensionality = u_quantity_matrix_A.shape[1]

    # DETERMINE TOTAL VECTORS i.e. NUMBER OF CUSTOMERS OR ROWS
    u_number_of_vectors = u_quantity_matrix_A.shape[0]

    # COMPUTE MATRIX RANK i.e. NUMBER OF LINEARLY INDEPENDENT COLUMNS
    u_matrix_rank = np.linalg.matrix_rank(u_quantity_matrix_A)

    # COMPUTE PSEUDO-INVERSE AND ESTIMATE PRODUCT PRICES i.e. SOLVE AX = C
    u_pseudo_inverse_A = np.linalg.pinv(u_quantity_matrix_A)
    u_price_vector_X = np.dot(u_pseudo_inverse_A, u_payment_vector_C)

    # PRINT FINAL OUTPUTS i.e. ANSWERS TO ALL 4 QUESTIONS
    print("📌 DIMENSIONALITY OF THE VECTOR SPACE i.e. NUMBER OF FEATURES:", u_dimensionality)
    print("📌 NUMBER OF VECTORS IN THE SPACE i.e. NUMBER OF CUSTOMERS:", u_number_of_vectors)
    print("📌 RANK OF MATRIX A i.e. INDEPENDENT FEATURES:", u_matrix_rank)
    print("📌 ESTIMATED COST PER PRODUCT [CANDIES, MANGOES, MILK]:", u_price_vector_X.flatten())

# INVOKE THE FUNCTION TO EXECUTE TASK A1
u_compute_product_prices_from_purchases()
