import pandas as pd

# Load your features CSV
ud_df_full = pd.read_csv("C:/Users/Udhaya/sem5_ML/features_lab3.csv")  # update path if needed

# Define the rule-based confidence labeling function
def assign_confidence(row):
    if row['silence_pct'] > 0.6 and row['rms'] < 0.01:
        return 1  # Very Low
    elif row['silence_pct'] > 0.4:
        return 2  # Low
    elif 0.3 < row['silence_pct'] <= 0.4:
        return 3  # Moderate
    elif row['silence_pct'] <= 0.3 and row['rms'] > 0.03:
        return 4  # High
    elif row['rms'] > 0.05 and row['silence_pct'] < 0.2:
        return 5  # Very High
    else:
        return 3  # Default to Moderate

# Apply the labeling to the DataFrame
ud_df_full['class'] = ud_df_full.apply(assign_confidence, axis=1)

# Save the labeled DataFrame
ud_df_full.to_csv("C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv", index=False)

print(" Class labels assigned and saved to 'features_lab3_labeled.csv'")
