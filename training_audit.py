#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tanítóadat modell-audit: per-minta WER és loss kiszámítása az AKTÍV Whisper modellel.

Cél: megmutatni, mely (akár már betanításra használt) minták viszik félre a tanulást.
- Magas WER a modell átirata vs. a tárolt referencia közt → valószínű hibás címke.
- Magas per-minta loss → a modell a tanítás ellenére sem tudja illeszteni (nehéz/hibás minta).

Az eredményt a training_data tábla audit_wer / audit_loss / audit_at oszlopaiba írja.
A training_quality.classify() ezeket beépíti az osztályozásba (⛔/⚠).

stdout (az app.py streameli a felületre):
  AUDIT: i/total id=ID wer=.. loss=..
  === Audit kész ===  /  === HIBA ... ===

Használat:
  python training_audit.py
"""
import os, sys, sqlite3
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from pydub import AudioSegment
from dotenv import load_dotenv

import training_quality

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)

CACHE_DIR         = Path(os.environ.get("HF_CACHE_DIR", str(BASE_DIR / "cache")))
MODEL_ID          = os.environ.get("WHISPER_MODEL_ID",  "Trendency/whisper-large-v3-hu")
MODEL_DIR         = os.environ.get("WHISPER_MODEL_DIR", "").strip()
MODEL_PATH        = MODEL_DIR if MODEL_DIR else MODEL_ID
DB_FILE           = BASE_DIR / "logs" / "transcriber.db"
TRAINING_DATA_DIR = BASE_DIR / "training_data"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def log(msg):
    print(msg, flush=True)


def load_samples():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    rows = c.execute("SELECT id, audio_fn, transcript FROM training_data ORDER BY id").fetchall()
    conn.close()
    return [{"id": r[0], "audio_fn": r[1], "transcript": r[2]} for r in rows]


def audio_to_array(path):
    seg = AudioSegment.from_file(str(path)).set_frame_rate(16000).set_channels(1)
    if len(seg) > 30000:
        seg = seg[:30000]
    return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0


def _resolve_audio_path(audio_fn):
    """A training_data.audio_fn lehet csak fájlnév vagy teljes út is."""
    p = Path(audio_fn)
    if p.is_absolute() and p.exists():
        return p
    return TRAINING_DATA_DIR / p.name


def main():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    samples = load_samples()
    if not samples:
        log("Nincs tanítóadat – nincs mit auditálni.")
        return True
    log(f"Auditálandó minták: {len(samples)} db")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    log(f"Eszköz: {device} ({dtype})")
    log(f"Modell betöltése: {MODEL_PATH}")

    processor = WhisperProcessor.from_pretrained(
        MODEL_PATH, cache_dir=str(CACHE_DIR), language="hu", task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, cache_dir=str(CACHE_DIR), torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()

    try:
        forced = processor.get_decoder_prompt_ids(language="hu", task="transcribe")
    except Exception:
        forced = None

    conn = sqlite3.connect(DB_FILE)
    now  = datetime.now().isoformat()
    total = len(samples)
    done = errors = 0

    for i, s in enumerate(samples, 1):
        path = _resolve_audio_path(s["audio_fn"])
        if not path.exists():
            log(f"AUDIT: {i}/{total} id={s['id']} HIÁNYZÓ FÁJL ({s['audio_fn']})")
            errors += 1
            continue
        try:
            arr   = audio_to_array(path)
            feats = processor.feature_extractor(
                arr, sampling_rate=16000, return_tensors="pt",
            ).input_features.to(device, dtype=dtype)

            # Per-minta loss (teacher forcing)
            labels = processor.tokenizer(
                s["transcript"], return_tensors="pt", max_length=448, truncation=True,
            ).input_ids.to(device)
            with torch.no_grad():
                loss = model(input_features=feats, labels=labels).loss.item()

            # WER: generálás → összevetés a referenciával
            with torch.no_grad():
                try:
                    gen = model.generate(input_features=feats, language="hu",
                                         task="transcribe", max_new_tokens=256)
                except TypeError:
                    gen = model.generate(input_features=feats, forced_decoder_ids=forced,
                                         max_new_tokens=256)
            hyp = processor.batch_decode(gen, skip_special_tokens=True)[0]
            wer = training_quality.wer(s["transcript"], hyp)

            conn.execute(
                "UPDATE training_data SET audit_wer=?, audit_loss=?, audit_at=? WHERE id=?",
                (round(float(wer), 4), round(float(loss), 4), now, s["id"]),
            )
            conn.commit()
            done += 1
            log(f"AUDIT: {i}/{total} id={s['id']} wer={wer*100:.0f}% loss={loss:.3f}")
        except Exception as e:
            errors += 1
            log(f"AUDIT: {i}/{total} id={s['id']} HIBA: {e}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    conn.close()
    log(f"Auditálva: {done}, hiba/kihagyva: {errors}")
    return True


if __name__ == "__main__":
    try:
        ok = main()
    except Exception as e:
        import traceback
        log(f"=== HIBA: {e} ===")
        traceback.print_exc()
        ok = False
    sys.exit(0 if ok else 1)
