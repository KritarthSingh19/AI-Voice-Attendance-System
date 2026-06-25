import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "ml/data/enrollment"
OUTPUT_CSV = "ml/features/features.csv"


def extract_mfcc(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean


all_features = []

for speaker in os.listdir(DATASET_PATH):

    speaker_path = os.path.join(DATASET_PATH, speaker)

    if not os.path.isdir(speaker_path):
        continue

    for audio_file in os.listdir(speaker_path):

        if not audio_file.endswith(".wav"):
            continue

        file_path = os.path.join(
            speaker_path,
            audio_file
        )

        features = extract_mfcc(file_path)

        row = {
            "speaker": speaker
        }

        for i, value in enumerate(features):
            row[f"mfcc_{i+1}"] = value

        all_features.append(row)

df = pd.DataFrame(all_features)

os.makedirs("ml/features", exist_ok=True)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nDataset Created Successfully\n")
print(df.head())
print(f"\nShape: {df.shape}")