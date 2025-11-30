from faster_whisper import WhisperModel
import json, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AUDIO_FILE = BASE_DIR / "audio.wav"
OUT_FILE = BASE_DIR / "templates" / "transcripts" / "fast_transcript.json"
os.makedirs(OUT_FILE.parent, exist_ok=True)

print("Fast Whisper modell betöltése...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

segments, _ = model.transcribe(str(AUDIO_FILE), beam_size=1, language="hu")

results = []
for seg in segments:
    results.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()})

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Fast Whisper kész: {OUT_FILE}")
