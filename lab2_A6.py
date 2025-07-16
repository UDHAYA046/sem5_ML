import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (update the filename if needed)
u_df = pd.read_excel("your_excel_file.xlsx", sheet_name="thyroid0387_UCI")

# Replace '?' with NaN and convert 'f'/'t' to 0/1
u_df = u_df.replace('?', np.nan)
u_df = u_df.replace({'f': 0, 't': 1})

# Drop non-numeric columns (like Record ID, sex, referral source, condition)
u_df_numeric = u_df.select_dtypes(include=[np.number])

# Drop rows with missing values (for clean cosine calculation)
u_df_numeric = u_df_numeric.dropna()

# Take the first 2 complete vectors
vec1 = u_df_numeric.iloc[0].values.reshape(1, -1)
vec2 = u_df_numeric.iloc[1].values.reshape(1, -1)

# Compute Cosine Similarity
cos_sim = cosine_similarity(vec1, vec2)[0][0]

print("Cosine Similarity between first two complete vectors:", round(cos_sim, 4))
