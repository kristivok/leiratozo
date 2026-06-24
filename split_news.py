#!/usr/bin/env python3
"""
split_news.py – Hírszegmens feldarabolása finomhangoláshoz

A szöveges fájl kétféle formátumot kezel:

  1. TÖBBBEKEZDÉSES (tipikus hírszöveg):
     Minden bekezdés (üres sorral elválasztva) = egy hírelem.
     A script N bekezdéshez N hangdarabot keres, a leghosszabb
     csendpontoknál vágva. Minden darabhoz pontosan egy bekezdés kerül.

  2. FOLYAMATOS SZÖVEG (egy bekezdés):
     A szöveget arányosan osztja a darabolási időpontokhoz.

Használat:
    python split_news.py hanganyag.mp3 hirszoveg.txt [kimeneti_mappa]

    Több fájl egyszerre:
    python split_news.py news/*.mp3 --txtdir news/ --out news/split/

Kimenet: kimeneti_mappa/
    hanganyag_001.wav + hanganyag_001.txt
    hanganyag_002.wav + hanganyag_002.txt  ...

Feltölthető a /finetune → Tömeges feltöltés funkcióval.
"""

import sys
import re
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_silence

MAX_CHUNK_MS   = 28_000   # 28s – Whisper 30s ablak alatt
MIN_CHUNK_MS   = 1_500    # 1.5s – ennél rövidebb darab kihagyva
SILENCE_DB     = -40      # dBFS – csendküszöb
MIN_SILENCE_MS = 200      # ms – minimális csend detektáláshoz


def load_audio(path: Path) -> AudioSegment:
    return AudioSegment.from_file(str(path)).set_frame_rate(16000).set_channels(1)


def parse_paragraphs(text: str) -> list[str]:
    parts = re.split(r'\n\s*\n', text.strip())
    return [re.sub(r'\s+', ' ', p).strip() for p in parts if p.strip()]


def find_all_silences(audio: AudioSegment) -> list[tuple[int, int]]:
    return detect_silence(audio, min_silence_len=MIN_SILENCE_MS, silence_thresh=SILENCE_DB)


def split_by_paragraph_count(audio: AudioSegment, n: int) -> list[tuple[int, int]]:
    """
    N bekezdéshez N hangdarabot keres a leghosszabb N-1 csendpont mentén.
    Ha nincs elég csend, arányosan osztja az időt.
    """
    total = len(audio)
    if n == 1:
        return [(0, total)]

    silences = find_all_silences(audio)

    if len(silences) >= n - 1:
        # Leghosszabb N-1 csend középpontjai lesznek a vágási pontok
        best = sorted(silences, key=lambda s: s[1] - s[0], reverse=True)[:n - 1]
        cut_points = sorted([(s + e) // 2 for s, e in best])
    else:
        # Nincs elég csend → egyenlő arányos vágás
        print(f"   FIGYELEM: csak {len(silences)} csendpont található, "
              f"{n - 1} kellene → arányos vágás")
        step = total // n
        cut_points = [step * i for i in range(1, n)]

    boundaries = [0] + cut_points + [total]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def split_proportional(audio: AudioSegment, text: str) -> list[tuple[tuple[int, int], str]]:
    """
    Folyamatos szöveget és a hangot arányosan darabolja MAX_CHUNK_MS-es darabokra.
    """
    total    = len(audio)
    words    = text.split()
    n_words  = len(words)
    silences = find_all_silences(audio)
    split_pts = sorted({0, total} | {(s + e) // 2 for s, e in silences})

    chunks = []
    pos = 0
    while pos < total:
        target = pos + MAX_CHUNK_MS
        if target >= total:
            chunks.append((pos, total))
            break
        best = max((p for p in split_pts if pos < p <= target), default=target)
        chunks.append((pos, best))
        pos = best

    # Szöveg arányos felosztása
    pairs = []
    for s, e in chunks:
        wi_s = round(s / total * n_words)
        wi_e = round(e / total * n_words)
        pairs.append(((s, e), ' '.join(words[wi_s:wi_e]).strip()))
    return pairs


def process_pair(audio_path: Path, text_path: Path, output_dir: Path) -> int:
    print(f"\n── {audio_path.name}")

    audio      = load_audio(audio_path)
    raw_text   = text_path.read_text(encoding='utf-8')
    paragraphs = parse_paragraphs(raw_text)
    total_ms   = len(audio)

    print(f"   Hossz: {total_ms / 1000:.1f}s | Bekezdések: {len(paragraphs)}")

    # ── Párok meghatározása ─────────────────────────────────────────────────
    if len(paragraphs) > 1:
        # Többbekezdéses: N csendpont mentén vágunk, bekezdésenként egy darab
        ranges = split_by_paragraph_count(audio, len(paragraphs))
        pairs  = list(zip(ranges, paragraphs))
    else:
        # Folyamatos szöveg: arányos darabolás
        single = paragraphs[0] if paragraphs else raw_text.strip()
        if total_ms <= MAX_CHUNK_MS:
            pairs = [((0, total_ms), single)]
        else:
            pairs = split_proportional(audio, single)

    # ── Mentés ─────────────────────────────────────────────────────────────
    kept  = 0
    stem  = audio_path.stem
    for i, ((s, e), chunk_text) in enumerate(pairs, 1):
        dur = (e - s) / 1000
        if not chunk_text.strip():
            print(f"   [{i:03d}] {dur:.1f}s – KIHAGYVA (üres szöveg)")
            continue
        if (e - s) < MIN_CHUNK_MS:
            print(f"   [{i:03d}] {dur:.1f}s – KIHAGYVA (túl rövid)")
            continue
        if (e - s) > MAX_CHUNK_MS:
            print(f"   [{i:03d}] {dur:.1f}s – FIGYELEM: meghaladja a {MAX_CHUNK_MS//1000}s határt")

        out_wav = output_dir / f"{stem}_{i:03d}.wav"
        out_txt = output_dir / f"{stem}_{i:03d}.txt"
        audio[s:e].export(str(out_wav), format='wav')
        out_txt.write_text(chunk_text, encoding='utf-8')
        print(f"   [{i:03d}] {dur:.1f}s | {len(chunk_text.split())} szó → {out_wav.name}")
        kept += 1

    return kept


def main():
    parser = argparse.ArgumentParser(
        description="Hírszegmens feldarabolása finomhangoláshoz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio",    nargs='+', help="Hangfájl(ok)")
    parser.add_argument("--txtdir", default=None,
                        help=".txt fájlok mappája (alapból: hangfájl mappája)")
    parser.add_argument("--out",    default=None,
                        help="Kimeneti mappa (alapból: split_output/)")
    args = parser.parse_args()

    audio_paths = [Path(p) for p in args.audio]
    missing = [p for p in audio_paths if not p.exists()]
    if missing:
        for m in missing:
            print(f"Nem található: {m}")
        sys.exit(1)

    total_chunks = 0
    for audio_path in audio_paths:
        txt_dir  = Path(args.txtdir) if args.txtdir else audio_path.parent
        txt_path = txt_dir / (audio_path.stem + '.txt')

        # Ha a szövegfájl neve hirszoveg.txt (egyedi eset: egy txt egy mp3-hoz)
        if not txt_path.exists():
            alt = audio_path.parent / 'hirszoveg.txt'
            if alt.exists():
                txt_path = alt
            else:
                print(f"KIHAGYVA (nincs leirat): {audio_path.name}")
                continue

        out_dir = Path(args.out) if args.out else audio_path.parent / "split_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        total_chunks += process_pair(audio_path, txt_path, out_dir)

    print(f"\n{'─' * 50}")
    print(f"Összesen {total_chunks} darab keletkezett.")
    print("Töltsd fel a /finetune oldalon → Tömeges feltöltés.")


if __name__ == '__main__':
    main()
