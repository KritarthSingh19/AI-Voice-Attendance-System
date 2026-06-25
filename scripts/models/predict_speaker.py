import librosa
import numpy as np
import joblib

model = joblib.load("ml/models/speaker_model.pkl")
encoder = joblib.load("ml/models/label_encoder.pkl")

audio_file = "ml/data/enrollment/kritarth_singh/sample_01.wav"

audio, sr = librosa.load(audio_file, sr=16000)

mfccs = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=13
)

feature_vector = np.mean(mfccs.T, axis=0)
feature_vector = feature_vector.reshape(1, -1)

import pandas as pd

columns = [f"mfcc_{i}" for i in range(1, 14)]

feature_df = pd.DataFrame(
    feature_vector,
    columns=columns
)

prediction = model.predict(feature_df)
speaker_name = encoder.inverse_transform(prediction)

print("\nPredicted Speaker:")
print(speaker_name[0])