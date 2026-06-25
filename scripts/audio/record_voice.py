import sounddevice as sd
import soundfile as sf
import os

SAMPLE_RATE = 16000
DURATION = 5  # seconds

def record_voice(save_path):
    print("🎤 Recording will start... Speak now!")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    sf.write(save_path, audio, SAMPLE_RATE)
    print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    name = input("Enter speaker name: ").strip().replace(" ", "_")

    save_dir = f"ml/data/enrollment/{name}"
    os.makedirs(save_dir, exist_ok=True)

    existing_files = [
        f for f in os.listdir(save_dir)
        if f.startswith("sample_") and f.endswith(".wav")
    ]

    next_sample_num = len(existing_files) + 1

    file_name = f"sample_{next_sample_num:02d}.wav"

    file_path = os.path.join(save_dir, file_name)

    record_voice(file_path)