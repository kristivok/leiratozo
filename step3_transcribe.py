#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, argparse

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def log(m): print(m, flush=True)

MODEL_ID  = os.environ.get("WHISPER_MODEL_ID", "Trendency/whisper-large-v3-hu")
MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "").strip()
MODEL_PATH = MODEL_DIR if MODEL_DIR else MODEL_ID
OFFLINE   = os.environ.get("HF_OFFLINE", "0") == "1"
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/srv/transcriber_app/cache")

PIPELINE_CHUNK_SEC = int(os.environ.get("PIPELINE_CHUNK_SEC", "20"))  # nagyobb = gyorsabb
NUM_BEAMS = 1                                # gyors és stabil
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "445"))

def _calc_allowed_max_new_tokens(model, processor, desired=445):
    max_target = getattr(getattr(model, "config", None), "max_target_positions", 448)
    try:
        forced = processor.get_decoder_prompt_ids(language="hu", task="transcribe")
        forced_len = len(forced) if isinstance(forced, list) else 0
    except Exception:
        forced_len = 0
    return max(1, min(desired, max_target - forced_len))

def _build_pipeline():
    cuda_ok = torch.cuda.is_available()
    dtype = torch.float16 if cuda_ok else torch.float32
    device_index = 0 if cuda_ok else -1
    log(f"CUDA elérhető: {cuda_ok}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=CACHE_DIR,
        local_files_only=OFFLINE,
        trust_remote_code=False,
        attn_implementation="eager",
    ).eval()
    if cuda_ok: model.to("cuda:0")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, cache_dir=CACHE_DIR, local_files_only=OFFLINE, trust_remote_code=False
    )
    allowed = _calc_allowed_max_new_tokens(model, processor, desired=MAX_NEW_TOKENS)

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=1,
        torch_dtype=dtype,
        device=device_index,
        chunk_length_s=PIPELINE_CHUNK_SEC,
        stride_length_s=(2, 1),
    )
    return asr, allowed

def _fast_available():
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False

def _transcribe_fw(chunk):
    from faster_whisper import WhisperModel
    model = WhisperModel(
        MODEL_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8",
    )
    seg_iter, _ = model.transcribe(
        chunk["file"], language="hu", beam_size=1, vad_filter=True, word_timestamps=True
    )
    text_parts, words = [], []
    for seg in seg_iter:
        if seg.text: text_parts.append(seg.text.strip())
        for w in (seg.words or []):
            words.append({"word": w.word, "start": chunk["start"] + float(w.start), "end": chunk["start"] + float(w.end)})
    return " ".join(text_parts).strip(), words

def _transcribe_hf(asr, allow_tokens, chunk, ts_mode):
    # ts_mode: 'off' | 'hf' | 'approx'
    if ts_mode == "hf":
        out = asr(
            chunk["file"],
            return_timestamps="word",
            generate_kwargs={
                "task": "transcribe", "language": "hu",
                "num_beams": NUM_BEAMS, "do_sample": False, "temperature": 0.0,
                "no_repeat_ngram_size": 3, "max_new_tokens": allow_tokens,
            },
        )
        text = (out.get("text") or "").strip()
        words = []
        for ch in out.get("chunks") or []:
            token = (ch.get("text") or "").strip()
            ts = ch.get("timestamp")
            if not token or not isinstance(ts, (list, tuple)): continue
            if ts[0] is None or ts[1] is None: continue
            try:
                ws, we = float(ts[0]), float(ts[1])
            except Exception:
                continue
            words.append({"word": token, "start": chunk["start"] + ws, "end": chunk["start"] + we})
        return text, words

    if ts_mode == "approx":
        out = asr(
            chunk["file"],
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe", "language": "hu",
                "num_beams": 1, "do_sample": False, "temperature": 0.0,
                "no_repeat_ngram_size": 3, "max_new_tokens": allow_tokens,
            },
        )
        text = (out.get("text") or "").strip()
        words = []
        for seg in out.get("chunks") or []:
            txt = (seg.get("text") or "").strip()
            ts = seg.get("timestamp")
            if not txt or not isinstance(ts, (list, tuple)): continue
            if ts[0] is None or ts[1] is None: continue
            try:
                seg_s, seg_e = float(ts[0]), float(ts[1])
            except Exception:
                continue
            toks = [w for w in txt.split() if w.strip()]
            if not toks: continue
            dur = max(1e-6, seg_e - seg_s)
            step = dur / len(toks)
            for i, w in enumerate(toks):
                ws = chunk["start"] + seg_s + i * step
                we = chunk["start"] + seg_s + (i + 1) * step
                words.append({"word": w, "start": ws, "end": we})
        return text, words

    # ts_mode == 'off'
    out = asr(
        chunk["file"],
        return_timestamps=False,
        generate_kwargs={
            "task": "transcribe", "language": "hu",
            "num_beams": NUM_BEAMS, "do_sample": False, "temperature": 0.0,
            "no_repeat_ngram_size": 3, "max_new_tokens": allow_tokens,
        },
    )
    return (out.get("text") or "").strip(), []

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--word-ts", choices=["off","hf","fw","approx"], default="off")
    args = ap.parse_args()

    chunks = json.load(open(args.chunks, "r", encoding="utf-8"))["chunks"]
    if not chunks:
        json.dump({"transcription": [], "words": []}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
        sys.exit(0)

    asr, allow_tokens = _build_pipeline()

    results, words_all = [], []
    for i, ch in enumerate(chunks, 1):
        fn = os.path.basename(ch["file"])
        log(f"[{i}/{len(chunks)}] Leiratozás: {fn} ({ch['speaker']})")
        t0 = time.time()
        try:
            if args.word_ts == "fw":
                if not _fast_available():
                    log("  -> faster-whisper nincs telepítve, vissza: HF (off)")
                    text, words = _transcribe_hf(asr, allow_tokens, ch, ts_mode="off")
                else:
                    text, words = _transcribe_fw(ch)
            else:
                text, words = _transcribe_hf(asr, allow_tokens, ch, ts_mode=args.word_ts)
        except Exception as e:
            log(f"  HF hiba ({e}), fallback: gyors off mód")
            text, words = _transcribe_hf(asr, allow_tokens, ch, ts_mode="off")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        log(f"  -> kész ({time.time()-t0:.2f}s), {len(text)} karakter, {len(words)} szó [{args.word_ts}]")
        results.append({"speaker": ch["speaker"], "start": ch["start"], "end": ch["end"], "text": text})
        words_all.extend([{"speaker": ch["speaker"], **w} for w in words])

    json.dump({"transcription": results, "words": words_all}, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log(f"ASR kimenet mentve: {args.out}")
