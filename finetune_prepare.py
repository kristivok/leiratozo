#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

# --- Konfiguráció ---
# Módosítsd ezeket az útvonalakat, ha szükséges.
# A script feltételezi, hogy a /srv/transcriber_app/fine_tune mappán belül fut.
BASE_DIR = Path("/srv/transcriber_app/fine_tune")
SOURCE_DIR = BASE_DIR / "00_source_audio"
PROCESSED_DIR = BASE_DIR / "01_processed_work"

# Azok a kiterjesztések, amiket hangfájlként kezelünk (lehet bővíteni)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}

def setup_directories():
    """
    Végignézi a SOURCE_DIR mappát, és minden új hangfájlhoz létrehozza
    a szükséges munkamappát és 'chunks' almappát a PROCESSED_DIR-ben.
    """
    print("--- Mappa előkészítő script indítása ---")
    
    # Biztosítjuk, hogy a kiindulási mappák léteznek
    try:
        SOURCE_DIR.mkdir(exist_ok=True)
        PROCESSED_DIR.mkdir(exist_ok=True)
    except OSError as e:
        print(f"HIBA: Nem sikerült létrehozni az alap mappákat: {e}")
        print("Ellenőrizd a jogosultságokat!")
        return

    print(f"Forrás mappa:   {SOURCE_DIR}")
    print(f"Cél mappa:      {PROCESSED_DIR}\n")

    processed_count = 0
    skipped_count = 0

    # Végigiterálunk a forrás mappában lévő összes elemen
    for audio_file in SOURCE_DIR.iterdir():
        # Csak a fájlokkal foglalkozunk, amiknek a kiterjesztése megfelel
        if audio_file.is_file() and audio_file.suffix.lower() in AUDIO_EXTENSIONS:
            
            # A mappa neve a fájlnév kiterjesztés nélkül
            dir_name = audio_file.stem
            target_dir = PROCESSED_DIR / dir_name
            
            # Ellenőrizzük, hogy a célmappa már létezik-e
            if target_dir.exists():
                # Ha igen, kihagyjuk
                skipped_count += 1
                # print(f"- Kihagyás: '{dir_name}' (mappa már létezik)")
                continue
            
            # Ha nem létezik, feldolgozzuk
            print(f"+ Feldolgozás: '{audio_file.name}'")
            processed_count += 1
            
            try:
                # Létrehozzuk a munkamappát és a 'chunks' almappát
                # A 'parents=True' biztosítja, hogy a teljes útvonal létrejöjjön
                chunks_dir = target_dir / "chunks"
                chunks_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"  - Létrehozva: {target_dir}")
                print(f"  - Létrehozva: {chunks_dir}")

            except OSError as e:
                print(f"  - HIBA: Nem sikerült létrehozni a mappát '{target_dir}'. Hiba: {e}")

    print("\n--- Összegzés ---")
    print(f"Újonnan feldolgozott fájlok: {processed_count}")
    print(f"Kihagyott (már létező) fájlok: {skipped_count}")
    print("Script befejezte a futást.")


if __name__ == "__main__":
    setup_directories()
