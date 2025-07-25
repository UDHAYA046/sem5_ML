import pandas as pd

# Load features
file_path = "C:/Users/Udhaya/sem5_ML/features_lab3.csv"
ud_df_full = pd.read_csv(file_path)

# Display value ranges for reference
print("Feature Value Ranges:")
for col in ['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']:
    print(f"{col:<15}: {ud_df_full[col].min():.6f} to {ud_df_full[col].max():.6f}")

# Rule-based labeling adjusted to actual range
def assign_confidence_class(row):
    mfcc1 = row['mfcc1']
    rms = row['rms']
    zcr = row['zcr']
    pitch_std = row['pitch_std']
    silence_pct = row['silence_pct']

    # Adjusted thresholds (data-driven)
    if (rms > 0.032 and pitch_std < 60 and silence_pct < 25 and zcr > 0.06 and mfcc1 > -380):
        return 5  # Very high confidence
    elif (rms > 0.028 and pitch_std < 65 and silence_pct < 30 and zcr > 0.055 and mfcc1 > -400):
        return 4  # High confidence
    elif (rms > 0.023 and pitch_std < 70 and silence_pct < 40 and zcr > 0.05 and mfcc1 > -420):
        return 3  # Moderate confidence
    elif (rms > 0.018 and pitch_std < 75 and silence_pct < 50 and zcr > 0.045):
        return 2  # Low confidence
    else:
        return 1  # Very low confidence

# Apply confidence labeling
ud_df_full['class'] = ud_df_full.apply(assign_confidence_class, axis=1)

# Display class distribution
print("\n Class labeling completed.")
print(ud_df_full['class'].value_counts())

# Save labeled file
output_path = "C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"
ud_df_full.to_csv(output_path, index=False)
print(f"\n Saved labeled file to: {output_path}")
