import os
import json
import torch
from dotenv import load_dotenv
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN nem található az .env fájlban")

# Fast Whisper uses the GPU in parallel, so diarization runs on CPU to avoid OOM.
# Set DIARIZATION_DEVICE=cuda in .env to override.
_device_str = os.getenv("DIARIZATION_DEVICE", "cpu")
DEVICE = torch.device(_device_str)
print(f"Használt eszköz: {DEVICE}")

pipeline = SpeakerDiarization.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)
pipeline.to(DEVICE)

AUDIO_FILE = BASE_DIR / "audio.wav"
if not Path(AUDIO_FILE).exists():
    raise FileNotFoundError(f"A fájl nem található: {AUDIO_FILE}")

batch_size = int(os.getenv("DIARIZATION_BATCH_SIZE", "8"))
pipeline._segmentation.batch_size = batch_size
print(f"Diarizáció futtatása (batch_size={batch_size})...")
diary = pipeline(AUDIO_FILE)

results = []
for turn, _, speaker in diary.itertracks(yield_label=True):
    results.append({
        "speaker": speaker,
        "start": round(turn.start, 3),
        "end": round(turn.end, 3)
    })

results.sort(key=lambda x: x["start"])

OUTPUT_JSON = BASE_DIR / "diarization_result.json"
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Diarizáció kész! Az eredmény mentve: {OUTPUT_JSON}")
print(f"Összes turn: {len(results)} db")
