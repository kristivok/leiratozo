#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse

def log(m): print(m, flush=True)

def merge_speaker_segments(diarization_data, max_gap=2.0):
    merged_segments, prev = [], None
    for seg in diarization_data:
        s = {"speaker": seg["speaker"], "start": float(seg["start"]), "end": float(seg["end"])}
        if prev and s["speaker"] == prev["speaker"] and (s["start"] - prev["end"]) <= max_gap:
            prev["end"] = max(prev["end"], s["end"])
        else:
            if prev: merged_segments.append(prev)
            prev = s
    if prev: merged_segments.append(prev)
    return merged_segments

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--max_gap", type=float, default=2.0)
    args = ap.parse_args()

    log(f"Diarizáció betöltése: {args.inp}")
    with open(args.inp, "r", encoding="utf-8") as f:
        diar = json.load(f)
    if not diar:
        log("Üres diarizáció!")
        sys.exit(1)

    diar = sorted(diar, key=lambda x: (x["start"], x["end"]))
    merged = merge_speaker_segments(diar, max_gap=args.max_gap)
    log(f"Egyesített szegmensek: {len(merged)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"segments": merged}, f, indent=2, ensure_ascii=False)
    log(f"Mentve: {args.out}")
