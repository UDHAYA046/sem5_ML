import librosa
import numpy as np
import pandas as pd
import os

input_folder = "C:/Users/Udhaya/sem5_ML/converted_wav/"
output_csv = "C:/Users/Udhaya/sem5_ML/features_lab3.csv"

data = []

for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        path = os.path.join(input_folder, file)
        y, sr = librosa.load(path, sr=None)

        # Feature 1: MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc1 = np.mean(mfccs[0])

        # Feature 2: RMS (volume)
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)

        # Feature 3: Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)

        # Feature 4: Pitch stability (using YIN)
        pitch = librosa.yin(y, fmin=50, fmax=300)
        pitch_std = np.std(pitch)

        # Feature 5: Silence %
        silence_segments = librosa.effects.split(y, top_db=20)
        silence_duration = sum([end - start for start, end in silence_segments])
        silence_pct = (silence_duration / len(y)) * 100

        data.append({
            "filename": file,
            "mfcc1": mfcc1,
            "rms": rms_mean,
            "zcr": zcr_mean,
            "pitch_std": pitch_std,
            "silence_pct": silence_pct
        })

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print(f"Feature extraction complete! Saved to: {output_csv}")
