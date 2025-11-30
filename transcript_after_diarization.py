import json
import os
import datetime
import time
import threading
import torch
from collections import defaultdict
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)
DEFAULT_CACHE = BASE_DIR / "cache"
CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", DEFAULT_CACHE))
os.environ.setdefault("HF_CACHE_DIR", str(CACHE_DIR))
DEFAULT_CHUNKS_DIR = BASE_DIR / "chunks"
TRANSCRIPTS_DIR = BASE_DIR / "templates" / "transcripts"
DIARIZATION_JSON = BASE_DIR / "diarization_result.json"
AUDIO_PATH = BASE_DIR / "audio.wav"

# ===== Alap beállítások =====
MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "Trendency/whisper-large-v3-hu")
MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "").strip()
MODEL_PATH = MODEL_DIR if MODEL_DIR else MODEL_ID

OFFLINE = os.environ.get("HF_OFFLINE", "0") == "1"

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if OFFLINE:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_flush_denormal(True)

def log(msg): print(msg, flush=True)

# ===== Heartbeat =====
class Heartbeat:
    def __init__(self, label:str, period:float=2.0):
        self.label = label
        self.period = period
        self._stop = threading.Event()
        self._t0 = time.time()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            elapsed = int(time.time() - self._t0)
            log(f"{self.label} folyamatban… {elapsed}s")
            self._stop.wait(self.period)

    def start(self): self._th.start()
    def stop(self):
        self._stop.set()
        self._th.join(timeout=1.0)

# ===== Diarizáció =====
def load_diarization_results(diarization_file):
    with open(diarization_file, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_and_sort_segments(diarization_data):
    norm = []
    for seg in diarization_data:
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

# ===== Audio szeletelés (turn-önként 1 wav) =====
def clean_chunks_directory(output_dir=DEFAULT_CHUNKS_DIR):
    output_dir = Path(output_dir)
    if output_dir.exists():
        for f in os.listdir(output_dir):
            p = output_dir / f
            try:
                if os.path.isfile(p): os.unlink(p)
            except Exception as e:
                log(f"Hiba a {p} törlésekor: {e}")
        log(f"A {output_dir} mappa tartalma törölve.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        log(f"A {output_dir} mappa létrehozva.")

def split_audio(audio_file, segments, output_dir=DEFAULT_CHUNKS_DIR):
    output_dir = Path(output_dir)
    clean_chunks_directory(output_dir)
    audio_file = Path(audio_file)
    if not audio_file.exists():
        raise FileNotFoundError(f"Nem található audio fájl: {audio_file}")
    audio = AudioSegment.from_wav(audio_file)
    chunk_files = []
    for i, seg in enumerate(segments):
        start, end, spk = seg["start"], seg["end"], seg["speaker"]
        chunk = audio[int(start*1000):int(end*1000)]
        fn = output_dir / f"chunk_{i}_{spk}.wav"
        chunk.export(str(fn), format="wav")
        chunk_files.append({"file": str(fn), "speaker": spk, "start": start, "end": end})
        log(f"Exportáltam: {fn.name} ({end-start:.1f}s, {spk})")
    log(f"Szeletelés kész. Turn-ök: {len(chunk_files)}")
    return chunk_files

# ===== ASR pipeline =====
def _load_trendency_pipeline():
    cuda_ok = torch.cuda.is_available()
    dtype = torch.float16 if cuda_ok else torch.float32
    device_index = 0 if cuda_ok else -1
    device_str = "cuda:0" if cuda_ok else "cpu"

    log(f"Trendency modell betöltése: {MODEL_PATH} | eszköz: {device_str}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        cache_dir=str(CACHE_DIR),
        local_files_only=OFFLINE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device_str).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        cache_dir=str(CACHE_DIR),
        local_files_only=OFFLINE,
    )

    # chunk_length_s=None -> saját időalapú chunkolást használunk
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device_index,
        chunk_length_s=None,
    )
    return asr, model, processor, device_str

# ===== Belső chunkolás egy turnon belül =====
def make_internal_chunks(turn_audio, chunk_size=25.0, overlap=3.0):
    """25 mp chunk + 3 mp overlap"""
    length = len(turn_audio) / 1000.0
    chunks = []
    start = 0.0

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append((start, end))
        start += (chunk_size - overlap)

    return chunks

# ===== Stitching =====
def stitch_texts(texts):
    """Chunkolt szövegek intelligens összefésülése."""
    if not texts:
        return ""

    final = texts[0].strip()

    for next_text in texts[1:]:
        next_text = next_text.strip()
        if not next_text:
            continue

        prev_words = final.split()[-12:]
        next_words = next_text.split()[:12]

        prev_join = " ".join(prev_words)
        next_join = " ".join(next_words)

        ratio = SequenceMatcher(None, prev_join, next_join).ratio()

        if ratio > 0.68:
            cut_point = len(prev_words)
            final = " ".join(final.split()[:-cut_point] + next_text.split())
        else:
            final = final + " " + next_text

    return " ".join(final.split())

# ===== Turn-önkénti leiratozás chunkolással =====
def transcribe_chunks(chunk_files):
    asr, model, processor, device_str = _load_trendency_pipeline()
    results = []

    try:
        total = len(chunk_files)
        for i, ch in enumerate(chunk_files, 1):
            log(f"[{i}/{total}] Leiratozás turnönként: {os.path.basename(ch['file'])} ({ch['speaker']})")

            turn_audio = AudioSegment.from_wav(ch["file"])
            internal_chunks = make_internal_chunks(turn_audio)

            turn_partial_texts = []

            for (cs, ce) in internal_chunks:
                part = turn_audio[int(cs*1000):int(ce*1000)]
                tmpfile = f"/tmp/_intchunk_{os.getpid()}_{time.time()}.wav"
                part.export(tmpfile, format="wav")

                out = asr(tmpfile, return_timestamps=False)
                if device_str.startswith("cuda"):
                    torch.cuda.synchronize()

                os.unlink(tmpfile)

                txt = (out.get("text", "") or "").strip()
                if txt:
                    turn_partial_texts.append(txt)

            final_text = stitch_texts(turn_partial_texts)

            results.append({
                "speaker": ch["speaker"],
                "start": ch["start"],
                "end": ch["end"],
                "text": final_text
            })

    finally:
        del asr, model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

# ===== Fast Whisper finomítás =====
def refine_with_fast_whisper(transcribed_segments, fast_json_path):
    fast_json_path = Path(fast_json_path)
    if not fast_json_path.exists():
        log("Fast Whisper referencia nem elérhető, finomítás kihagyva.")
        return transcribed_segments

    try:
        with open(fast_json_path, "r", encoding="utf-8") as f:
            fast_data = json.load(f)
    except Exception as e:
        log(f"Fast Whisper JSON betöltési hiba, finomítás kihagyva: {e}")
        return transcribed_segments

    full_sentences = []
    for s in fast_data:
        t = (s.get("text") or "").strip()
        if t:
            parts = [p.strip() for p in t.replace("?", ".").replace("!", ".").split(".") if p.strip()]
            full_sentences.extend(parts)

    if not full_sentences:
        log("Fast Whisper referencia üres, finomítás kihagyva.")
        return transcribed_segments

    for seg in transcribed_segments:
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        matches = get_close_matches(t, full_sentences, n=1, cutoff=0.6)
        if matches:
            seg["text"] = matches[0]

    return transcribed_segments

def summarize_speaker_times(segments):
    d = defaultdict(float)
    for s in segments:
        d[s["speaker"]] += (s["end"] - s["start"])
    return {k: round(v, 3) for k, v in d.items()}

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

# ===== Fő =====
if __name__ == "__main__":
    diarization_file = DIARIZATION_JSON
    audio_file = AUDIO_PATH
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = TRANSCRIPTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / f"final_transcription_{timestamp}.json"

    log("--- Folyamat indítása ---")
    log("1) Diarizációs adatok betöltése és rendezése...")
    diarization_data = load_diarization_results(diarization_file)
    if not diarization_data:
        log("ÜRES diarizáció! Leállok.")
        raise SystemExit(1)

    sorted_segments = normalize_and_sort_segments(diarization_data)
    log(f"Eredeti turn-ök száma: {len(sorted_segments)}")

    merged_turns = merge_consecutive_same_speaker(sorted_segments, max_gap=1.0)
    log(f"Összefűzött turn-ök: {len(merged_turns)}")

    log("2) Szeletelés turn-önként...")
    chunk_files = split_audio(audio_file, merged_turns, output_dir=DEFAULT_CHUNKS_DIR)
    if not chunk_files:
        log("Nincs mit leiratozni (0 turn). Leállok.")
        raise SystemExit(1)

    log("3) Leiratozás chunkokkal (Trendency modell)...")
    transcription_results = transcribe_chunks(chunk_files)

    fast_ref = output_dir / "fast_transcript.json"
    log("3/b) Fast Whisper kontextus szerinti finomítás...")
    transcription_results = refine_with_fast_whisper(transcription_results, fast_ref)

    log("4) Összegzés és mentés...")
    speaker_summary = summarize_speaker_times(merged_turns)
    final_output = {"summary": speaker_summary, "transcription": transcription_results}
    save_json(output_file, final_output)

    latest_json = output_dir / "latest_final.json"
    save_json(latest_json, final_output)

    clean_text = "\n".join([seg["text"].strip() for seg in transcription_results if seg["text"].strip()])
    clean_output_file = output_dir / f"final_text_{timestamp}.txt"
    with open(clean_output_file, "w", encoding="utf-8") as f:
        f.write(clean_text)

    log(f"A leiratozott beszélgetés mentve: {output_file}")
