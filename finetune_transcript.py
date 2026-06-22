#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import whisper
import csv
from pathlib import Path

# --- Konfiguráció ---
BASE_DIR = Path("/srv/transcriber_app/fine_tune")
PROCESSED_DIR = BASE_DIR / "01_processed_work"
# Új mappa az ellenőrizendő CSV fájloknak
REVIEW_DIR = BASE_DIR / "02_to_be_reviewed"

# Whisper Modell beállításai
MODEL_SIZE = "large-v3" # Választható: "tiny", "base", "small", "medium", "large-v3"
LANGUAGE = "hu"
# Opcionális "súgó" a Whispernek a nevek és szakkifejezések javítására
# Hagyd üresen (""), ha nem szeretnéd használni.
INITIAL_PROMPT = "ATV, Németh Sándor, Rónai Egon, Pintér Sándor, Orbán Viktor, Fidesz, DK, MSZP, költségvetés, belügyminisztérium."

def main():
    """
    Végigiterál a feldolgozott munkamappákon, leiratozza a chunkokat
    és létrehozza az ellenőrzésre váró CSV fájlokat.
    """
    print("--- Piszkozat Átirat Készítő Script Indítása ---")
    
    # Biztosítjuk, hogy a kimeneti mappa létezik
    REVIEW_DIR.mkdir(exist_ok=True)
    
    # --- Whisper Modell Betöltése (CSAK EGYSZER) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Használt eszköz: {device}")
    print(f"Whisper modell ({MODEL_SIZE}) betöltése...")
    try:
        model = whisper.load_model(MODEL_SIZE, device=device)
        print("Modell sikeresen betöltve.")
    except Exception as e:
        print(f"HIBA: A Whisper modell betöltése sikertelen: {e}")
        return

    # --- Feldolgozási Ciklus ---
    for work_dir in PROCESSED_DIR.iterdir():
        if not work_dir.is_dir():
            continue

        print(f"\n--- Ellenőrzés: {work_dir.name} ---")
        
        chunks_dir = work_dir / "chunks"
        output_csv_path = REVIEW_DIR / f"{work_dir.name}.csv"
        
        # 1. Ellenőrizzük, hogy a kimeneti CSV már létezik-e
        if output_csv_path.exists():
            print("  - Kihagyás: Az ellenőrizendő CSV fájl már létezik.")
            continue
            
        # 2. Ellenőrizzük, hogy van-e 'chunks' mappa
        if not chunks_dir.exists() or not chunks_dir.is_dir():
            print(f"  - Kihagyás: Nem található 'chunks' mappa itt: {work_dir}")
            continue

        # 3. Gyűjtsük össze a leiratozandó chunkokat
        # A sorted biztosítja, hogy a CSV sorrendje mindig ugyanaz legyen.
        chunk_files = sorted(chunks_dir.glob("*.wav"))
        if not chunk_files:
            print(f"  - Kihagyás: Nincsenek .wav fájlok a 'chunks' mappában.")
            continue
        
        print(f"  - {len(chunk_files)} chunk leiratozása indul a '{REVIEW_DIR.name}' mappába...")
        
        transcription_results = []
        try:
            # 4. Chunkok leiratozása egyenként
            for i, chunk_path in enumerate(chunk_files, 1):
                print(f"    ({i}/{len(chunk_files)}) -> {chunk_path.name}")
                
                result = model.transcribe(
                    str(chunk_path),
                    language=LANGUAGE,
                    initial_prompt=INITIAL_PROMPT,
                    # fp16=False # Ha pontossági problémák vannak, próbáld ezt a sort aktiválni
                )
                
                # Relatív útvonal a CSV-be
                relative_path = f"{work_dir.name}/chunks/{chunk_path.name}"
                
                # Eredmény hozzáadása a listához
                transcription_results.append({
                    "file_name": relative_path,
                    "transcription": result["text"].strip()
                })
        
        except Exception as e:
            print(f"  - HIBA a leiratozás közben: {e}")
            continue # Ugrás a következő munkamappára

        # 5. Eredmények mentése CSV fájlba
        if transcription_results:
            print(f"  - Leiratozás kész. Eredmények mentése ide: {output_csv_path}")
            try:
                with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['file_name', 'transcription']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    writer.writerows(transcription_results)
            except IOError as e:
                print(f"  - HIBA a CSV fájl írása közben: {e}")

    print("\n--- Script befejezte a futást ---")


if __name__ == "__main__":
    main()
