#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diarizáció + mondathatáros darabolás finomhangolási tanítóadat-készítéshez.

Workflow:
  1. Diarizáció (pyannote) – cachelve, ha diarization_result.json már létezik
  2. Átírás faster-whisper szó-timestampekkel
  3. Szavak hozzárendelése bemondóhoz (diarizáció alapján)
  4. Mondathatáros csomagolás ~30s darabokba
  5. Export: chunk_NNNN_SPEAKER.wav + chunk_NNNN_SPEAKER.txt (javítható)
  6. manifest.json – batch_import.py-hoz

Használat:
  python diar_sentence_split.py --audio beszed.mp3 --out tananyag/

Javítás után importálás:
  python diar_batch_import.py --dir tananyag/
"""

import os
import sys
import json
import argparse
import re
import torch
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
_model_dir = os.getenv("WHISPER_MODEL_DIR", "").strip()
_model_id  = os.getenv("WHISPER_MODEL_ID", "Trendency/whisper-large-v3-hu")
# WHISPER_MODEL_DIR a LoRA fine-tune kimenet is lehet (PyTorch formátum),
# amit a faster-whisper nem tud olvasni (model.bin kell). Ilyenkor az eredeti ID-t használjuk.
if _model_dir and Path(_model_dir, "model.bin").exists():
    MODEL_PATH = _model_dir
else:
    MODEL_PATH = _model_id
CACHE_DIR = os.getenv("HF_CACHE_DIR", str(BASE_DIR / "cache"))

TARGET_MIN_S = 15.0   # rövidebb mint ez: folytassuk a következő mondattal
TARGET_MAX_S = 35.0   # ha elértük: mindenképp vágjuk (ha van mondatvég)
HARD_MAX_S   = 40.0   # abszolút max, mondathatártól függetlenül

SENTENCE_END = re.compile(r'[.!?…]\s*$')


def log(msg):
    print(msg, flush=True)


# ── Diarizáció ────────────────────────────────────────────────────────────────

def run_diarization(wav_path: Path, cache_json: Path) -> list[dict]:
    if cache_json.exists():
        log(f"Diarizáció betöltve cache-ből: {cache_json}")
        with open(cache_json, encoding="utf-8") as f:
            return json.load(f)

    if not HUGGINGFACE_TOKEN:
        log("HIBA: HUGGINGFACE_TOKEN nincs beállítva az .env-ben.")
        sys.exit(1)

    from pyannote.audio import Pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Pyannote betöltése ({device})...")
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    pipe.to(device)

    log("Diarizáció futtatása...")
    diary = pipe(str(wav_path))

    segments = [
        {"speaker": spk, "start": round(turn.start, 3), "end": round(turn.end, 3)}
        for turn, _, spk in diary.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda x: x["start"])

    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    log(f"Diarizáció kész: {len(segments)} turn, mentve: {cache_json}")
    return segments


# ── Átfedő időszakok azonosítása ──────────────────────────────────────────────

def find_overlaps(segments: list[dict]) -> list[tuple[float, float]]:
    """Visszaadja az időintervallumokat ahol 2+ bemondó aktív egyszerre."""
    events = []
    for seg in segments:
        events.append((seg["start"], +1))
        events.append((seg["end"],   -1))
    events.sort()

    overlaps = []
    active = 0
    overlap_start = None
    for t, delta in events:
        was_overlap = active >= 2
        active += delta
        is_overlap = active >= 2
        if not was_overlap and is_overlap:
            overlap_start = t
        elif was_overlap and not is_overlap and overlap_start is not None:
            overlaps.append((overlap_start, t))
            overlap_start = None
    return overlaps


def in_overlap(t_start: float, t_end: float, overlaps: list[tuple[float, float]]) -> bool:
    mid = (t_start + t_end) / 2
    for os_, oe in overlaps:
        if os_ <= mid <= oe:
            return True
    return False


# ── Átírás HuggingFace transformers pipeline-nal ─────────────────────────────

def transcribe_words(wav_path: Path) -> list[dict]:
    """Visszaad [{word, start, end}, ...] listát szó-timestampekkel."""
    log(f"Átírás HF transformers pipeline-nal: {MODEL_PATH}")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

    cuda_ok = torch.cuda.is_available()
    dtype    = torch.float16 if cuda_ok else torch.float32
    device_i = 0 if cuda_ok else -1

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=CACHE_DIR,
        trust_remote_code=False,
        attn_implementation="eager",
    ).eval()
    if cuda_ok:
        model.to("cuda:0")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, cache_dir=CACHE_DIR, trust_remote_code=False
    )

    # Maximálisan engedélyezett új tokenek kiszámítása
    max_target = getattr(getattr(model, "config", None), "max_target_positions", 448)
    try:
        forced = processor.get_decoder_prompt_ids(language="hu", task="transcribe")
        forced_len = len(forced) if isinstance(forced, list) else 0
    except Exception:
        forced_len = 0
    max_new_tokens = max(1, min(445, max_target - forced_len))

    asr = hf_pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=1,
        torch_dtype=dtype,
        device=device_i,
        chunk_length_s=20,
        stride_length_s=(2, 1),
    )

    # return_timestamps=True → szegmens-szintű timestampek (stabil hosszú audión)
    # A szavakat arányosan osztjuk el a szegmensen belül – elég pontosságú a daraboláshoz.
    out = asr(
        str(wav_path),
        return_timestamps=True,
        generate_kwargs={
            "task": "transcribe", "language": "hu",
            "num_beams": 1, "do_sample": False, "temperature": 0.0,
            "no_repeat_ngram_size": 3, "max_new_tokens": max_new_tokens,
        },
    )

    words = []
    for seg in out.get("chunks") or []:
        txt = (seg.get("text") or "").strip()
        ts  = seg.get("timestamp")
        if not txt or not isinstance(ts, (list, tuple)):
            continue
        if ts[0] is None:
            continue
        try:
            seg_s = float(ts[0])
            seg_e = float(ts[1]) if ts[1] is not None else seg_s + 5.0
        except Exception:
            continue
        toks = [w for w in txt.split() if w.strip()]
        if not toks:
            continue
        dur  = max(0.01, seg_e - seg_s)
        step = dur / len(toks)
        for i, w in enumerate(toks):
            words.append({
                "word":  w,
                "start": seg_s + i * step,
                "end":   seg_s + (i + 1) * step,
            })

    log(f"Átírás kész: {len(words)} szó ({len(out.get('chunks', []))} szegmens)")
    return words


# ── Szó → bemondó hozzárendelés ───────────────────────────────────────────────

def assign_speakers(words: list[dict], segments: list[dict]) -> list[dict]:
    """Minden szóhoz hozzárendeli a legtöbbet fedő diarizációs bemondót."""
    result = []
    for w in words:
        best_spk = "UNKNOWN"
        best_overlap = 0.0
        for seg in segments:
            ol = min(w["end"], seg["end"]) - max(w["start"], seg["start"])
            if ol > best_overlap:
                best_overlap = ol
                best_spk = seg["speaker"]
        result.append({**w, "speaker": best_spk})
    return result


# ── Mondathatáros csomagolás ~30s-os darabokba ───────────────────────────────

def is_sentence_end(word: str, next_word: str | None, gap: float) -> bool:
    if SENTENCE_END.search(word):
        return True
    if gap > 1.5:
        return True
    return False


def pack_chunks(
    words: list[dict],
    overlaps: list[tuple[float, float]],
    skip_overlap: bool,
) -> list[dict]:
    """
    Mondathatáros csomagolás.
    Visszaad [{speaker, start, end, words: [...]}, ...] listát.
    """
    chunks = []
    buf_words = []
    buf_start = None
    buf_speaker = None

    def flush(buf_words, buf_start, buf_speaker):
        if not buf_words:
            return None
        text = " ".join(w["word"] for w in buf_words).strip()
        return {
            "speaker": buf_speaker,
            "start": buf_start,
            "end": buf_words[-1]["end"],
            "text": text,
            "word_count": len(buf_words),
        }

    for i, w in enumerate(words):
        # Átfedő részt kihagyjuk ha kérték
        if skip_overlap and in_overlap(w["start"], w["end"], overlaps):
            if buf_words:
                ch = flush(buf_words, buf_start, buf_speaker)
                if ch:
                    chunks.append(ch)
                buf_words, buf_start, buf_speaker = [], None, None
            continue

        # Bemondóváltás → mindenképp új chunk
        if buf_speaker and w["speaker"] != buf_speaker and buf_words:
            ch = flush(buf_words, buf_start, buf_speaker)
            if ch:
                chunks.append(ch)
            buf_words, buf_start, buf_speaker = [], None, None

        if not buf_words:
            buf_start = w["start"]
            buf_speaker = w["speaker"]

        buf_words.append(w)
        dur = w["end"] - buf_start

        # Következő szó gapje
        next_w = words[i + 1] if i + 1 < len(words) else None
        gap = (next_w["start"] - w["end"]) if next_w else 99.0

        sent_end = is_sentence_end(w["word"], next_w["word"] if next_w else None, gap)

        if dur >= HARD_MAX_S:
            # Abszolút max elérve
            ch = flush(buf_words, buf_start, buf_speaker)
            if ch:
                chunks.append(ch)
            buf_words, buf_start, buf_speaker = [], None, None
        elif dur >= TARGET_MAX_S and sent_end:
            ch = flush(buf_words, buf_start, buf_speaker)
            if ch:
                chunks.append(ch)
            buf_words, buf_start, buf_speaker = [], None, None
        elif dur >= TARGET_MIN_S and sent_end:
            ch = flush(buf_words, buf_start, buf_speaker)
            if ch:
                chunks.append(ch)
            buf_words, buf_start, buf_speaker = [], None, None

    if buf_words:
        ch = flush(buf_words, buf_start, buf_speaker)
        if ch:
            chunks.append(ch)

    return chunks


# ── Export ────────────────────────────────────────────────────────────────────

def export_chunks(
    audio: AudioSegment,
    chunks: list[dict],
    out_dir: Path,
    pad_ms: int = 100,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, ch in enumerate(chunks):
        spk = ch["speaker"].replace(" ", "_")
        base = f"chunk_{i:04d}_{spk}"
        wav_path = out_dir / f"{base}.wav"
        txt_path = out_dir / f"{base}.txt"

        s_ms = max(0, int(ch["start"] * 1000) - pad_ms)
        e_ms = min(len(audio), int(ch["end"] * 1000) + pad_ms)
        segment = audio[s_ms:e_ms]

        # 16kHz mono WAV – ez a Whisper fine-tune optimális bemenete
        segment.set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ch["text"] + "\n")

        dur = (e_ms - s_ms) / 1000.0
        manifest.append({
            "wav": str(wav_path),
            "txt": str(txt_path),
            "speaker": ch["speaker"],
            "start": ch["start"],
            "end": ch["end"],
            "duration_s": round(dur, 3),
            "text": ch["text"],
        })
        log(f"  [{i+1:04d}] {base}.wav  {dur:.1f}s  \"{ch['text'][:60]}...\"" if len(ch['text']) > 60 else
            f"  [{i+1:04d}] {base}.wav  {dur:.1f}s  \"{ch['text']}\"")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"\nManifest mentve: {manifest_path}")
    return manifest


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global TARGET_MIN_S, TARGET_MAX_S

    ap = argparse.ArgumentParser(description="Diarizáció + mondathatáros darabolás fine-tune tanítóadathoz")
    ap.add_argument("--audio",         required=True, help="Bemeneti hangfájl (mp3/wav/m4a/...)")
    ap.add_argument("--out",           default="diar_chunks", help="Kimeneti mappa (default: diar_chunks/)")
    ap.add_argument("--diar-cache",    help="Diarizációs JSON cache elérési útja (default: <out>/diarization_result.json)")
    ap.add_argument("--skip-overlap",  action="store_true", default=True,
                    help="Átfedő bemondók szakaszait kihagyja (default: be)")
    ap.add_argument("--keep-overlap",  action="store_true",
                    help="Átfedő szakaszokat is belerakja (felülírja --skip-overlap)")
    ap.add_argument("--target-min",    type=float, default=TARGET_MIN_S, help=f"Min célhossz s-ban (default: {TARGET_MIN_S})")
    ap.add_argument("--target-max",    type=float, default=TARGET_MAX_S, help=f"Max célhossz s-ban (default: {TARGET_MAX_S})")
    ap.add_argument("--pad-ms",        type=int, default=100, help="Padding ms (default: 100)")
    args = ap.parse_args()

    TARGET_MIN_S = args.target_min
    TARGET_MAX_S = args.target_max
    skip_overlap = args.skip_overlap and not args.keep_overlap

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        log(f"HIBA: A fájl nem létezik: {audio_path}")
        sys.exit(1)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    diar_cache = Path(args.diar_cache).resolve() if args.diar_cache else out_dir / "diarization_result.json"

    log(f"\n=== Diarizáció + mondathatáros darabolás ===")
    log(f"Bemeneti fájl: {audio_path}")
    log(f"Kimeneti mappa: {out_dir}")
    log(f"Átfedő szakaszok: {'kihagyva' if skip_overlap else 'benne maradnak'}")
    log(f"Célhossz: {TARGET_MIN_S}–{TARGET_MAX_S}s\n")

    # 1. Normalizálás (16kHz mono WAV)
    log("Hangfájl betöltése és normalizálása...")
    audio = AudioSegment.from_file(str(audio_path))
    wav_path = out_dir / "_normalized.wav"
    audio.set_frame_rate(16000).set_channels(1).export(str(wav_path), format="wav")
    log(f"Normalizálva: {wav_path}  ({len(audio)/1000:.1f}s)")

    # 2. Diarizáció
    diar_segments = run_diarization(wav_path, diar_cache)

    speakers = list({s["speaker"] for s in diar_segments})
    log(f"Azonosított bemondók: {speakers}")

    # 3. Átfedő szakaszok
    overlaps = find_overlaps(diar_segments)
    overlap_total = sum(e - s for s, e in overlaps)
    log(f"Átfedő szakaszok: {len(overlaps)} db, összesen {overlap_total:.1f}s")

    # 4. Átírás szó-timestampekkel
    words = transcribe_words(wav_path)
    if not words:
        log("HIBA: Nincsenek szavak az átírásban.")
        sys.exit(1)

    # 5. Bemondó hozzárendelés
    words = assign_speakers(words, diar_segments)

    # 6. Mondathatáros csomagolás
    log("\nMondathatáros csomagolás...")
    chunks = pack_chunks(words, overlaps, skip_overlap)
    log(f"Létrehozott chunk-ok: {len(chunks)}")

    total_dur = sum(ch["end"] - ch["start"] for ch in chunks)
    log(f"Összes tanítóadat: {total_dur/60:.1f} perc")

    # 7. Export
    log("\nExportálás...")
    manifest = export_chunks(audio, chunks, out_dir, pad_ms=args.pad_ms)

    # 8. Statisztika
    durations = [m["duration_s"] for m in manifest]
    log(f"\n=== Összefoglaló ===")
    log(f"Chunkok száma:     {len(manifest)}")
    log(f"Össz tanítóadat:   {sum(durations)/60:.1f} perc")
    log(f"Átlagos hossz:     {sum(durations)/len(durations):.1f}s" if durations else "")
    log(f"Kihagyott overlap: {overlap_total:.1f}s")
    log(f"\nKövetkezó lépés: javítsd a .txt fájlokat, majd:")
    log(f"  python diar_batch_import.py --dir {out_dir}")

    # Normalizált WAV törlése (csak segédfájl volt)
    wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
