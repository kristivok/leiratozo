#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, argparse
from pydub import AudioSegment
from pydub.utils import which

def log(m): print(m, flush=True)

def export_range(audio: AudioSegment, s: float, e: float, out_path: str) -> bool:
    if e - s < 0.2:  # <200 ms -> kihagy
        return False
    audio[int(s*1000):int(e*1000)].export(out_path, format="wav")
    return True

def split_audio(audio_file: str, segments: list, out_dir: str,
                max_len_s: float, overlap_s: float, max_subchunks_per_segment: int):
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Nincs ilyen audio: {audio_file}")
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            try: os.unlink(os.path.join(out_dir, f))
            except: pass
        log(f"A {out_dir} mappa tartalma törölve.")
    else:
        os.makedirs(out_dir, exist_ok=True)
        log(f"A {out_dir} mappa létrehozva.")

    log(f"FFmpeg: {which('ffmpeg') or 'N/A'}, FFprobe: {which('ffprobe') or 'N/A'}")
    audio = AudioSegment.from_wav(audio_file)

    total_len = sum(max(0.0, s["end"] - s["start"]) for s in segments)
    log(f"Szeletelés indul. Szegmensek: {len(segments)}, teljes hossz ~{total_len:.1f}s")

    chunks = []
    exported = 0
    for idx, seg in enumerate(segments):
        start, end, spk = float(seg["start"]), float(seg["end"]), seg["speaker"]
        if end - start <= 0.0:
            continue

        cur = start
        sub_i = 0
        guard = 0
        while cur < end - 1e-6:
            s = cur
            e = min(end, s + max_len_s)
            fn = os.path.join(out_dir, f"chunk_{idx}_{sub_i}_{spk}.wav")
            t0 = time.time()
            if export_range(audio, s, e, fn):
                exported += 1
                log(f"Exportáltam al-szeletet [{exported}]: {os.path.basename(fn)} "
                    f"({s:.2f}-{e:.2f}s, dur={e-s:.2f}s)")
                chunks.append({"file": fn, "speaker": spk, "start": s, "end": e})

            if e >= end - 1e-6:
                break  # VÉGE – nincs további overlap

            step = (e - s) - overlap_s
            if step <= 1e-3:
                step = (e - s)  # biztos előrelépés
            cur = min(end, s + step)
            if cur <= s + 1e-6:
                cur = e  # vészfék
            sub_i += 1
            guard += 1
            if guard > max_subchunks_per_segment:
                log(f"FIGYELEM: túl sok al-szelet egy szegmensnél (>{max_subchunks_per_segment}) – megszakítom ennél.")
                break

    log(f"Szeletelés kész. Létrejött chunkok: {len(chunks)} db")
    return chunks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_dir", default="chunks")
    ap.add_argument("--max_len_s", type=float, default=float(os.environ.get("MAX_SLICE_SEC", "28.5")))
    ap.add_argument("--overlap_s", type=float, default=float(os.environ.get("SLICE_OVERLAP_SEC", "1.0")))
    ap.add_argument("--limit_per_seg", type=int, default=int(os.environ.get("MAX_SUBCHUNKS_PER_SEG", "5000")))
    args = ap.parse_args()

    with open(args.segments, "r", encoding="utf-8") as f:
        merged = json.load(f)["segments"]

    chunks = split_audio(args.audio, merged, args.chunk_dir, args.max_len_s, args.overlap_s, args.limit_per_seg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, indent=2, ensure_ascii=False)
    log(f"Mentve: {args.out}")
