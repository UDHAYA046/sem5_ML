from pydub import AudioSegment
import os

# Your actual paths
input_folder = "C:/Users/Udhaya/sem5_ML/VivaData_Set2_23012/VivaData_Set2_23012/"
output_folder = "C:/Users/Udhaya/sem5_ML/converted_wav/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert all .mp3 files to .wav
for file in os.listdir(input_folder):
    if file.endswith(".mp3"):
        mp3_path = os.path.join(input_folder, file)
        wav_path = os.path.join(output_folder, file.replace(".mp3", ".wav"))
        
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        print(f" Converted: {file} → {wav_path}")
