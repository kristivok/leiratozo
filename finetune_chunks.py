#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment, exceptions as pydub_exceptions

# --- Konfiguráció ---
BASE_DIR = Path("/srv/transcriber_app/fine_tune")
SOURCE_DIR = BASE_DIR / "00_source_audio"
PROCESSED_DIR = BASE_DIR / "01_processed_work"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}

# --- Hugging Face és Eszköz Beállítása ---
try:
    load_dotenv()
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if not HUGGINGFACE_TOKEN:
        raise ValueError("HUGGINGFACE_TOKEN nem található az .env fájlban vagy a környezeti változók között.")
except ImportError:
    raise ImportError("A 'python-dotenv' csomag nincs telepítve. Telepítsd: pip install python-dotenv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_source_audio_file(work_dir_name: str) -> Path | None:
    """Megkeresi a forrás hangfájlt a work dir neve alapján."""
    for ext in AUDIO_EXTENSIONS:
        source_file = SOURCE_DIR / f"{work_dir_name}{ext}"
        if source_file.exists():
            return source_file
    return None

def convert_to_temp_wav(source_path: Path) -> Path | None:
    """
    Bemeneti hangfájlt ideiglenes, 16kHz-es mono WAV fájllá konvertál.
    Ez stabilizálja a pyannote feldolgozást. Visszatér az ideiglenes fájl útvonalával.
    """
    print(f"  - Normalizálás: '{source_path.name}' konvertálása ideiglenes WAV-ra...")
    try:
        audio = AudioSegment.from_file(source_path)
        # Létrehozunk egy ideiglenes fájlt a rendszer temp mappájában
        temp_wav_path = Path(tempfile.gettempdir()) / f"temp_{source_path.stem}.wav"
        
        # Konvertálás 16kHz mono WAV-ra, ami a legtöbb STT modellnek ideális
        audio.set_frame_rate(16000).set_channels(1).export(temp_wav_path, format="wav")
        
        print(f"  - Ideiglenes fájl létrehozva: {temp_wav_path}")
        return temp_wav_path
    except pydub_exceptions.CouldntDecodeError:
        print(f"  - HIBA: A pydub nem tudta dekódolni a fájlt: {source_path.name}. Lehet, hogy sérült.")
        return None
    except Exception as e:
        print(f"  - HIBA a WAV konverzió során: {e}")
        return None


def chunk_audio_from_diarization(source_audio_path: Path, diarization_json_path: Path, chunks_output_dir: Path):
    """A diarizációs JSON alapján feldarabolja az EREDETI hangfájlt."""
    print("  - Hangfájl darabolása (chunking) indul...")
    try:
        audio = AudioSegment.from_file(source_audio_path)
        with open(diarization_json_path, 'r', encoding='utf-8') as f:
            diarization_data = json.load(f)

        for i, segment in enumerate(diarization_data):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            speaker = segment["speaker"]
            
            chunk_audio = audio[start_ms:end_ms]
            
            chunk_filename = f"chunk_{i:04d}_{speaker}.wav"
            chunk_filepath = chunks_output_dir / chunk_filename
            
            chunk_audio.export(chunk_filepath, format="wav")
        
        print(f"  - Darabolás kész. {len(diarization_data)} chunk mentve ide: {chunks_output_dir}")

    except Exception as e:
        print(f"  - HIBA a darabolás során: {e}")


def main():
    """
    Fő feldolgozási ciklus. Iterál a munkamappákon, elvégzi a diarizációt és a darabolást.
    """
    print("--- Diarizációs és Daraboló Script Indítása ---")
    print(f"Használt eszköz: {DEVICE}")

    try:
        print("Pyannote diarizációs pipeline betöltése...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN
        )
        diarization_pipeline.to(DEVICE)
        print("Pipeline sikeresen betöltve.")
    except Exception as e:
        print(f"HIBA: A Pyannote pipeline betöltése sikertelen: {e}")
        return

    for work_dir in PROCESSED_DIR.iterdir():
        if not work_dir.is_dir():
            continue

        print(f"\n--- Ellenőrzés: {work_dir.name} ---")
        diarization_json_path = work_dir / "diarization_result.json"
        
        if diarization_json_path.exists():
            print("  - Kihagyás: 'diarization_result.json' már létezik.")
            continue

        source_audio_path = find_source_audio_file(work_dir.name)
        if not source_audio_path:
            print(f"  - HIBA: Nem található forrás hangfájl '{work_dir.name}' névvel a '{SOURCE_DIR}' mappában.")
            continue
        
        print(f"  - Forrásfájl megtalálva: {source_audio_path}")
        
        temp_wav_path = None
        try:
            # 1. Konvertálás ideiglenes WAV-ra a stabil feldolgozáshoz
            temp_wav_path = convert_to_temp_wav(source_audio_path)
            if not temp_wav_path:
                # Ha a konverzió sikertelen, kihagyjuk a fájlt
                continue

            # 2. Diarizáció futtatása az ideiglenes, tiszta WAV fájlon
            print("  - Diarizáció futtatása...")
            diary = diarization_pipeline(str(temp_wav_path), num_speakers=None)

            # 3. Eredmény JSON formátumra hozása és mentése
            result = [
                {"speaker": speaker, "start": round(turn.start, 3), "end": round(turn.end, 3)}
                for turn, _, speaker in diary.itertracks(yield_label=True)
            ]
            
            with open(diarization_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"  - Diarizáció kész! Eredmény mentve ide: {diarization_json_path}")
            
            # 4. Darabolás a frissen létrehozott JSON és az EREDETI fájl alapján
            chunks_dir = work_dir / "chunks"
            chunks_dir.mkdir(exist_ok=True)
            chunk_audio_from_diarization(source_audio_path, diarization_json_path, chunks_dir)

        except Exception as e:
            print(f"  - VÉGZETES HIBA a '{work_dir.name}' feldolgozása során: {e}")
        
        finally:
            # 5. Az ideiglenes WAV fájl törlése, akár sikeres volt a futás, akár nem
            if temp_wav_path and temp_wav_path.exists():
                temp_wav_path.unlink()
                print(f"  - Ideiglenes fájl törölve: {temp_wav_path}")

    print("\n--- Script befejezte a futást ---")


if __name__ == "__main__":
    main()
