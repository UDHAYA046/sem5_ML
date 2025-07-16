import pandas as pd

def u_calculate_jc_smc():
    """
    CALCULATES JACCARD COEFFICIENT AND SIMPLE MATCHING COEFFICIENT
    BETWEEN FIRST TWO OBSERVATION VECTORS USING BINARY ATTRIBUTES ONLY.
    """

    # LOAD DATA
    u_df = pd.read_excel(
        r"C:\Users\Udhaya\sem5_ML\Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI"
    )
    u_df.columns = u_df.columns.str.strip()

    # REPLACE '?' WITH NaN AND DROP NA COLUMNS FROM FIRST TWO ROWS
    pd.set_option('future.no_silent_downcasting', True)
    u_df = u_df.replace('?', pd.NA)
    u_binary_cols = [col for col in u_df.columns if set(u_df[col].dropna().unique()) <= {'f', 't'}]

    # EXTRACT FIRST TWO BINARY VECTORS
    u_vec1 = u_df.loc[0, u_binary_cols].replace({'f': 0, 't': 1}).astype(int)
    u_vec2 = u_df.loc[1, u_binary_cols].replace({'f': 0, 't': 1}).astype(int)

    # COUNT MATCHES
    f11 = ((u_vec1 == 1) & (u_vec2 == 1)).sum()
    f00 = ((u_vec1 == 0) & (u_vec2 == 0)).sum()
    f10 = ((u_vec1 == 1) & (u_vec2 == 0)).sum()
    f01 = ((u_vec1 == 0) & (u_vec2 == 1)).sum()

    # CALCULATE JC AND SMC
    u_jaccard = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
    u_smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) != 0 else 0

    # DISPLAY OUTPUT
    print("BINARY ATTRIBUTES USED:", len(u_binary_cols))
    print("f11 =", f11, "  f00 =", f00, "  f10 =", f10, "  f01 =", f01)
    print("Jaccard Coefficient (JC):", round(u_jaccard, 4))
    print("Simple Matching Coefficient (SMC):", round(u_smc, 4))

# CALL FUNCTION
u_calculate_jc_smc()
