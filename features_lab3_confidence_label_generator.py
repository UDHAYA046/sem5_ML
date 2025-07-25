import pandas as pd

# Load the CSV
file_path = "C:/Users/Udhaya/sem5_ML/features_lab3.csv"  # Adjust path if needed
ud_df_full = pd.read_csv(file_path)

# Print feature ranges for debugging
print("Feature Value Ranges:")
print("mfcc1       :", ud_df_full['mfcc1'].min(), "to", ud_df_full['mfcc1'].max())
print("rms         :", ud_df_full['rms'].min(), "to", ud_df_full['rms'].max())
print("zcr         :", ud_df_full['zcr'].min(), "to", ud_df_full['zcr'].max())
print("pitch_std   :", ud_df_full['pitch_std'].min(), "to", ud_df_full['pitch_std'].max())
print("silence_pct :", ud_df_full['silence_pct'].min(), "to", ud_df_full['silence_pct'].max())

# Define rule-based confidence classification function
def assign_confidence(row):
    mfcc1 = row['mfcc1']
    rms = row['rms']
    zcr = row['zcr']
    pitch_std = row['pitch_std']
    silence_pct = row['silence_pct']

    # Adjusted thresholds based on realistic feature values
    if rms > 0.012 and pitch_std < 5 and silence_pct < 10 and zcr > 0.05 and mfcc1 > -500:
        return 5  # Very High Confidence
    elif rms > 0.01 and pitch_std < 8 and silence_pct < 20 and zcr > 0.045 and mfcc1 > -600:
        return 4  # High Confidence
    elif rms > 0.008 and pitch_std < 12 and silence_pct < 35 and zcr > 0.04 and mfcc1 > -700:
        return 3  # Moderate Confidence
    elif rms > 0.005 and pitch_std < 18 and silence_pct < 50 and zcr > 0.02:
        return 2  # Low Confidence
    else:
        return 1  # Very Low Confidence

# Apply the function to label the data
ud_df_full['class'] = ud_df_full.apply(assign_confidence, axis=1)

# Save to new CSV
output_path = "C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"
ud_df_full.to_csv(output_path, index=False)

# Display result counts
print("\n✅ Class labeling completed.")
print(ud_df_full['class'].value_counts())
print(f"\nSaved labeled file to: {output_path}")
