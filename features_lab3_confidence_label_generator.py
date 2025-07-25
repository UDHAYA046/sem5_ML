import pandas as pd

# Load your features CSV
ud_df_full = pd.read_csv("C:/Users/Udhaya/sem5_ML/features_lab3.csv")  # Update path if needed

# Define the rule-based confidence labeling function
def assign_confidence(row):
    mfcc1 = row['mfcc1']
    rms = row['rms']
    zcr = row['zcr']
    pitch_std = row['pitch_std']
    silence_pct = row['silence_pct']

    # Class 5: Very High Confidence
    if (rms > 0.04 and pitch_std < 15 and silence_pct < 5 and zcr > 0.09 and mfcc1 > -200):
        return 5

    # Class 4: High Confidence
    elif (rms > 0.03 and pitch_std < 25 and silence_pct < 10 and zcr > 0.08 and mfcc1 > -300):
        return 4

    # Class 3: Moderate Confidence
    elif (rms > 0.025 and pitch_std < 35 and silence_pct < 20 and zcr > 0.07 and mfcc1 > -400):
        return 3

    # Class 2: Low Confidence
    elif (rms > 0.015 and pitch_std < 45 and silence_pct < 30 and zcr > 0.05):
        return 2

    # Class 1: Very Low Confidence (fallback case)
    else:
        return 1

# Apply the labeling function
ud_df_full['class'] = ud_df_full.apply(assign_confidence, axis=1)

# Save the labeled DataFrame to a new CSV
output_path = "C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"
ud_df_full.to_csv(output_path, index=False)

print(f" Class labels assigned and saved to '{output_path}'")
