#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch import: a diar_sentence_split.py által létrehozott, kézzel javított
.txt fájlokat betölti a training_data SQLite adatbázisba és bemásolja
a WAV fájlokat a training_data/ mappába.

Használat:
  python diar_batch_import.py --dir tananyag/
  python diar_batch_import.py --dir tananyag/ --dry-run    # csak listáz, nem ír DB-be
  python diar_batch_import.py --dir tananyag/ --min-dur 3  # min 3 másodperces chunkok
"""

import os
import sys
import json
import sqlite3
import shutil
import argparse
from datetime import datetime
from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent
DB_FILE         = BASE_DIR / "logs" / "transcriber.db"
TRAINING_DIR    = BASE_DIR / "training_data"


def log(msg):
    print(msg, flush=True)


def load_manifest(dir_path: Path) -> list[dict]:
    manifest_path = dir_path / "manifest.json"
    if not manifest_path.exists():
        log(f"HIBA: manifest.json nem található: {manifest_path}")
        sys.exit(1)
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def read_corrected_text(txt_path: Path) -> str:
    """Visszaadja a kézzel javított szöveget a .txt fájlból."""
    if not txt_path.exists():
        return ""
    return txt_path.read_text(encoding="utf-8").strip()


def import_samples(
    manifest: list[dict],
    dir_path: Path,
    min_dur: float,
    dry_run: bool,
) -> tuple[int, int, int]:
    TRAINING_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_FILE) if not dry_run else None
    if conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS training_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_fn    TEXT,
            transcript  TEXT,
            uploaded_at TEXT,
            duration_s  REAL,
            used_in_run INTEGER DEFAULT 0
        )""")
        conn.commit()

    imported = skipped_short = skipped_empty = 0
    now = datetime.now().isoformat()

    for item in manifest:
        wav_src = Path(item["wav"])
        txt_src = Path(item["txt"])
        dur     = item["duration_s"]

        if dur < min_dur:
            log(f"  KIHAGYVA (túl rövid {dur:.1f}s): {wav_src.name}")
            skipped_short += 1
            continue

        text = read_corrected_text(txt_src)
        if not text:
            log(f"  KIHAGYVA (üres szöveg): {txt_src.name}")
            skipped_empty += 1
            continue

        dest_wav = TRAINING_DIR / wav_src.name
        # Ha már létezik ilyen nevű fájl, egyedi nevet adunk
        if dest_wav.exists():
            stem = wav_src.stem
            suffix = wav_src.suffix
            counter = 1
            while dest_wav.exists():
                dest_wav = TRAINING_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

        if dry_run:
            log(f"  [DRY] {wav_src.name} ({dur:.1f}s)  \"{text[:70]}\"")
        else:
            shutil.copy2(wav_src, dest_wav)
            conn.execute(
                "INSERT INTO training_data (audio_fn, transcript, uploaded_at, duration_s) VALUES (?,?,?,?)",
                (str(dest_wav), text, now, dur),
            )
            conn.commit()
            log(f"  OK {dest_wav.name} ({dur:.1f}s)  \"{text[:70]}\"")

        imported += 1

    if conn:
        conn.close()

    return imported, skipped_short, skipped_empty


def main():
    ap = argparse.ArgumentParser(description="Batch import javított tanítóadatokból")
    ap.add_argument("--dir",     required=True, help="A diar_sentence_split.py kimeneti mappája")
    ap.add_argument("--min-dur", type=float, default=2.0, help="Min. hossz másodpercben (default: 2.0)")
    ap.add_argument("--dry-run", action="store_true", help="Csak listáz, nem módosít DB-t")
    args = ap.parse_args()

    dir_path = Path(args.dir).resolve()
    if not dir_path.exists():
        log(f"HIBA: A mappa nem létezik: {dir_path}")
        sys.exit(1)

    log(f"\n=== Batch import ===")
    log(f"Forrás mappa:  {dir_path}")
    log(f"DB:            {DB_FILE}")
    log(f"Cél WAV mappa: {TRAINING_DIR}")
    log(f"Min. hossz:    {args.min_dur}s")
    if args.dry_run:
        log("MODUS: dry-run (nem ír DB-be)\n")

    manifest = load_manifest(dir_path)
    log(f"Manifest: {len(manifest)} chunk\n")

    imported, skipped_short, skipped_empty = import_samples(
        manifest, dir_path, args.min_dur, args.dry_run
    )

    log(f"\n=== Összefoglaló ===")
    log(f"Importált:       {imported}")
    log(f"Kihagyva (rövid): {skipped_short}")
    log(f"Kihagyva (üres):  {skipped_empty}")

    if not args.dry_run and imported > 0:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT COUNT(*), COALESCE(SUM(duration_s),0) FROM training_data")
        total_n, total_dur = c.fetchone()
        conn.close()
        log(f"\nAdatbázis mostantól: {total_n} minta, {total_dur/60:.1f} perc")


if __name__ == "__main__":
    main()
