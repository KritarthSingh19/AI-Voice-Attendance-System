import os
import tempfile

import joblib
import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
DURATION = 5


def verify_voice():

    print("\nRecording voice for verification...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp_file:

        temp_path = temp_file.name

    sf.write(
        temp_path,
        audio,
        SAMPLE_RATE
    )

    model = joblib.load(
        "ml/models/speaker_model.pkl"
    )

    encoder = joblib.load(
        "ml/models/label_encoder.pkl"
    )

    audio_data, sr = librosa.load(
        temp_path,
        sr=16000
    )

    mfccs = librosa.feature.mfcc(
        y=audio_data,
        sr=sr,
        n_mfcc=13
    )

    feature_vector = np.mean(
        mfccs.T,
        axis=0
    )

    feature_vector = feature_vector.reshape(
        1,
        -1
    )

    columns = [
        f"mfcc_{i}"
        for i in range(1, 14)
    ]

    feature_df = pd.DataFrame(
        feature_vector,
        columns=columns
    )

    prediction = model.predict(
        feature_df
    )

    speaker_name = encoder.inverse_transform(
        prediction
    )[0]

    os.remove(temp_path)

    return speaker_name