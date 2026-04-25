import os
import soundfile as sf
from datasets import load_dataset

print("Downloading Vietnamese cohort from Fleurs via HuggingFace...")
os.makedirs("data/cohort", exist_ok=True)

ds = load_dataset("PolyAI/minds14", name="en-US", split="train", streaming=True)

count = 0
for item in ds:
    audio = item["audio"]["array"]
    sr = item["audio"]["sampling_rate"]
    filename = f"data/cohort/vi_{count:03d}.wav"
    sf.write(filename, audio, sr)
    count += 1
    if count >= 50:
        break

print(f"Downloaded {count} samples to data/cohort.")
