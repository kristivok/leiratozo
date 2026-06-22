import json
import os
import datetime
import numpy as np
import torch
from collections import defaultdict
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)

DEFAULT_CACHE = BASE_DIR / "cache"
CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", DEFAULT_CACHE))
os.environ.setdefault("HF_CACHE_DIR", str(CACHE_DIR))

TRANSCRIPTS_DIR = BASE_DIR / "templates" / "transcripts"
DIARIZATION_JSON = BASE_DIR / "diarization_result.json"
AUDIO_PATH = BASE_DIR / "audio.wav"

MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "Trendency/whisper-large-v3-hu")
MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "").strip()
MODEL_PATH = MODEL_DIR if MODEL_DIR else MODEL_ID
OFFLINE = os.environ.get("HF_OFFLINE", "0") == "1"

# ASR_BATCH_SIZE: hány audio chunk kerül egy GPU batch-be.
# num_beams=1 (greedy) esetén: RTX 4070 Ti SUPER (16 GB) → 8 biztonságos,
# kisebb GPU-nál (pl. 8 GB) → 2-4.
ASR_BATCH_SIZE = int(os.environ.get("ASR_BATCH_SIZE", "8"))

# Greedy dekódolás (num_beams=1): a beam search-hez képest töredék VRAM
# (beam_size × batch mérete × szekvencia = sokszoros memória).
# A Trendency modell alapból beam search-t futtat, ezért felül kell írni.
_num_beams = int(os.environ.get("ASR_NUM_BEAMS", "5"))
GENERATE_KWARGS = {
    "task": "transcribe",
    "language": "hu",
    "num_beams": _num_beams,
    "do_sample": False,
    "temperature": 0.0,
    "no_repeat_ngram_size": 3,
    "max_new_tokens": 444,
}

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if OFFLINE:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_flush_denormal(True)


def log(msg): print(msg, flush=True)


# ── Diarizáció feldolgozása ───────────────────────────────────────────────────

def load_diarization_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_and_sort_segments(data):
    norm = []
    for seg in data:
        try:
            spk = str(seg["speaker"])
            st = float(seg["start"])
            en = float(seg["end"])
            if en - st > 0.05:
                norm.append({"speaker": spk, "start": round(st, 3), "end": round(en, 3)})
        except Exception as e:
            log(f"Hibás diarizációs elem kihagyva: {seg} | {e}")
    norm.sort(key=lambda s: (s["start"], s["end"]))
    return norm


def merge_consecutive_same_speaker(segments, max_gap=1.0):
    """Azonos szomszédos speaker turn-öket összefűz, ha a köz <= max_gap mp."""
    if not segments:
        return []
    merged = []
    cur = segments[0].copy()
    for s in segments[1:]:
        if s["speaker"] == cur["speaker"] and (s["start"] - cur["end"]) <= max_gap:
            cur["end"] = max(cur["end"], s["end"])
        else:
            merged.append(cur)
            cur = s.copy()
    merged.append(cur)
    return merged


# ── ASR pipeline ─────────────────────────────────────────────────────────────

def _load_trendency_pipeline():
    cuda_ok = torch.cuda.is_available()
    dtype = torch.float16 if cuda_ok else torch.float32
    device_index = 0 if cuda_ok else -1
    device_str = "cuda:0" if cuda_ok else "cpu"

    log(f"Trendency modell betöltése: {MODEL_PATH} | {device_str} | batch={ASR_BATCH_SIZE}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=str(CACHE_DIR),
        local_files_only=OFFLINE,
        attn_implementation="eager",
    ).to(device_str).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, cache_dir=str(CACHE_DIR), local_files_only=OFFLINE,
    )

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device_index,
        # chunk_length_s: Whisper max 30s – hosszú turn-öket automatikusan darabolja
        chunk_length_s=30,
        stride_length_s=1,
        batch_size=ASR_BATCH_SIZE,
    )
    return asr, model, processor


def transcribe_turns(turns, audio_file):
    """
    Az összes speaker turn-t batch-ben átírja.
    Az audio.wav-ot egyszer tölti be memóriába – nincs per-turn disk I/O.
    """
    log(f"Audio betöltése memóriába: {audio_file}")
    full_audio = AudioSegment.from_wav(str(audio_file))

    log(f"Turn-ök numpy array-be konvertálása ({len(turns)} db)...")
    inputs = []
    for turn in turns:
        start_ms = int(turn["start"] * 1000)
        end_ms = int(turn["end"] * 1000)
        chunk = full_audio[start_ms:end_ms]
        samples = np.array(chunk.get_array_of_samples(), dtype=np.float32) / 32768.0
        inputs.append({"array": samples, "sampling_rate": 16000})

    asr, model, processor = _load_trendency_pipeline()
    log(f"Batch ASR futtatása ({len(inputs)} turn, batch_size={ASR_BATCH_SIZE}, num_beams={_num_beams})...")
    outputs = []
    try:
        for i in range(0, len(inputs), ASR_BATCH_SIZE):
            batch = inputs[i:i + ASR_BATCH_SIZE]
            batch_out = list(asr(batch, return_timestamps=False, generate_kwargs=GENERATE_KWARGS))
            outputs.extend(batch_out)
            done = min(i + ASR_BATCH_SIZE, len(inputs))
            print(f"PROGRESS: {done}/{len(inputs)}", flush=True)
            log(f"  {done}/{len(inputs)} turn kész")
    finally:
        del asr, model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log("GPU cache törölve.")

    results = []
    for turn, out in zip(turns, outputs):
        results.append({
            "speaker": turn["speaker"],
            "start": turn["start"],
            "end": turn["end"],
            "text": (out.get("text", "") or "").strip(),
        })
    return results


# ── Mentés ───────────────────────────────────────────────────────────────────

def summarize_speaker_times(segments):
    d = defaultdict(float)
    for s in segments:
        d[s["speaker"]] += (s["end"] - s["start"])
    return {k: round(v, 3) for k, v in d.items()}


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


# ── Belépési pont ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    output_file = TRANSCRIPTS_DIR / f"final_transcription_{timestamp}.json"

    log("--- Leiratozás indul ---")

    log("1) Diarizációs adatok betöltése...")
    diarization_data = load_diarization_results(DIARIZATION_JSON)
    if not diarization_data:
        log("ÜRES diarizáció! Leállok.")
        raise SystemExit(1)

    sorted_segments = normalize_and_sort_segments(diarization_data)
    log(f"   Szegmensek: {len(sorted_segments)}")

    merged_turns = merge_consecutive_same_speaker(sorted_segments, max_gap=1.0)
    log(f"   Összefűzött turn-ök: {len(merged_turns)}")

    if not merged_turns:
        log("Nincs leiratozható turn. Leállok.")
        raise SystemExit(1)

    log("2) Batch ASR (Trendency modell, GPU)...")
    transcription_results = transcribe_turns(merged_turns, AUDIO_PATH)

    log("3) Mentés...")
    speaker_summary = summarize_speaker_times(merged_turns)
    final_output = {"summary": speaker_summary, "transcription": transcription_results}
    save_json(output_file, final_output)

    latest_json = TRANSCRIPTS_DIR / "latest_final.json"
    save_json(latest_json, final_output)

    clean_text = "\n".join(
        seg["text"].strip()
        for seg in transcription_results
        if seg["text"].strip()
    )
    clean_output_file = TRANSCRIPTS_DIR / f"final_text_{timestamp}.txt"
    with open(clean_output_file, "w", encoding="utf-8") as f:
        f.write(clean_text)

    log(f"A leiratozott beszélgetés mentve: {output_file}")
