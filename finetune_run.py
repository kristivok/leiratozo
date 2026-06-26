#!/usr/bin/env python3
"""
Whisper LoRA fine-tuning script.
Subprocess-ként hívja az app.py schedulere vagy manuális trigger.

Kimenet (stdout):
  STEP: current/total loss=X.XXXX
  EPOCH: X/Y avg_loss=X.XXXX
  === Kész ===  /  === HIBA ... ===
"""
import os, sys, argparse, sqlite3
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from pydub import AudioSegment
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)

CACHE_DIR         = Path(os.environ.get("HF_CACHE_DIR",    str(BASE_DIR / "cache")))
MODEL_ID          = os.environ.get("WHISPER_MODEL_ID",     "Trendency/whisper-large-v3-hu")
MODEL_DIR         = os.environ.get("WHISPER_MODEL_DIR",    "").strip()
MODEL_PATH        = MODEL_DIR if MODEL_DIR else MODEL_ID
DB_FILE           = BASE_DIR / "logs" / "transcriber.db"
TRAINING_DATA_DIR = BASE_DIR / "training_data"
FINETUNE_OUTPUT   = BASE_DIR / "finetune_output"

LORA_RANK  = int(os.environ.get("FINETUNE_LORA_RANK",   "32"))
LORA_ALPHA = int(os.environ.get("FINETUNE_LORA_ALPHA",  "64"))
EPOCHS     = int(os.environ.get("FINETUNE_EPOCHS",      "3"))
BATCH_SIZE = int(os.environ.get("FINETUNE_BATCH_SIZE",  "4"))
GRAD_ACCUM = int(os.environ.get("FINETUNE_GRAD_ACCUM",  "1"))
LR         = float(os.environ.get("FINETUNE_LR",        "1e-4"))
MAX_STEPS  = int(os.environ.get("FINETUNE_MAX_STEPS",   "0"))  # 0 = korlátlan


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def log(msg):
    print(msg, flush=True)


def load_samples():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, audio_fn, transcript FROM training_data ORDER BY id")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "audio_fn": r[1], "transcript": r[2]} for r in rows]


def mark_used(sample_ids, run_id):
    conn = sqlite3.connect(DB_FILE)
    ph = ",".join("?" * len(sample_ids))
    conn.execute(f"UPDATE training_data SET used_in_run=? WHERE id IN ({ph})",
                 [run_id] + list(sample_ids))
    conn.commit()
    conn.close()


def audio_to_array(path):
    seg = (AudioSegment.from_file(str(path))
           .set_frame_rate(16000).set_channels(1))
    if len(seg) > 30000:
        seg = seg[:30000]
    return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0


def run_finetune(run_id: int):
    from transformers import (
        WhisperForConditionalGeneration, WhisperProcessor,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset, DataLoader

    log(f"=== Finomhangolás indul (run_id={run_id}) ===")
    log(f"Idő: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    samples = load_samples()
    if not samples:
        log("Nincs felhasználható tanítóadat – leállok.")
        return False
    log(f"Tanítóminták: {len(samples)} db")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    log(f"Eszköz: {device} ({dtype})")

    log(f"Modell betöltése: {MODEL_PATH}")
    processor = WhisperProcessor.from_pretrained(
        MODEL_PATH, cache_dir=str(CACHE_DIR), language="hu", task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, cache_dir=str(CACHE_DIR),
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )

    log(f"LoRA alkalmazása (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # Encoder lefagyasztása PEFT után — LoRA paramétereket is beleértve.
    # Így a backprop csak a decoder LoRA súlyain fut át.
    frozen = 0
    for name, param in model.named_parameters():
        if ".encoder." in name:
            param.requires_grad_(False)
            frozen += 1
    log(f"Encoder lefagyasztva ({frozen} param, csak decoder LoRA tanítható).")

    model.to(device)
    torch.cuda.empty_cache()

    trainable, total = model.get_nb_trainable_parameters()
    log(f"Tanítható paraméterek: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    log(f"GPU memória betöltés után: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    model.train()

    # ── Dataset ──────────────────────────────────────────────────────────────

    class _DS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            s   = self.items[i]
            arr = audio_to_array(TRAINING_DATA_DIR / s["audio_fn"])
            feats = processor.feature_extractor(
                arr, sampling_rate=16000, return_tensors="pt",
            ).input_features[0]
            labels = processor.tokenizer(
                s["transcript"], return_tensors="pt",
                max_length=448, truncation=True,
            ).input_ids[0]
            return {"input_features": feats, "labels": labels}

    def _collate(batch):
        feats = torch.stack([b["input_features"] for b in batch])
        max_l = max(b["labels"].shape[0] for b in batch)
        padded = torch.full((len(batch), max_l), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            padded[i, :b["labels"].shape[0]] = b["labels"]
        return {"input_features": feats, "labels": padded}

    loader = DataLoader(_DS(samples), batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=_collate, num_workers=0)
    # Optimizer lépések száma (mini-batch / GRAD_ACCUM)
    opt_steps_per_epoch = max(1, len(loader) // GRAD_ACCUM)
    total_steps = EPOCHS * opt_steps_per_epoch
    if MAX_STEPS > 0:
        total_steps = min(total_steps, MAX_STEPS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(10, max(1, total_steps // 10)),
        num_training_steps=total_steps,
    )

    log(f"Tanítás: {EPOCHS} epoch, batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM} "
        f"(grad_accum={GRAD_ACCUM}), lr={LR}, opt_steps={total_steps}")

    global_step = 0
    last_loss   = None
    optimizer.zero_grad()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss   = 0.0
        epoch_steps  = 0
        accum_loss   = 0.0
        mini_step    = 0

        for batch in loader:
            if MAX_STEPS > 0 and global_step >= MAX_STEPS:
                break

            inp    = batch["input_features"].to(device, dtype=dtype)
            labels = batch["labels"].to(device)

            loss = model(input_features=inp, labels=labels).loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()
            mini_step  += 1

            if mini_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                sched.step()
                optimizer.zero_grad()

                last_loss    = accum_loss
                epoch_loss  += last_loss
                epoch_steps += 1
                global_step += 1
                accum_loss   = 0.0
                log(f"STEP: {global_step}/{total_steps} loss={last_loss:.4f}")

        avg = epoch_loss / max(epoch_steps, 1)
        log(f"EPOCH: {epoch}/{EPOCHS} avg_loss={avg:.4f}")

    # ── Mentés ────────────────────────────────────────────────────────────────
    log(f"Modell összefűzése és mentése: {FINETUNE_OUTPUT}")
    os.makedirs(FINETUNE_OUTPUT, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(FINETUNE_OUTPUT))
    processor.save_pretrained(str(FINETUNE_OUTPUT))

    mark_used([s["id"] for s in samples], run_id)
    log(f"=== Kész === loss={last_loss:.4f}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()
    try:
        ok = run_finetune(args.run_id)
    except Exception as e:
        import traceback
        log(f"=== HIBA: {e} ===")
        traceback.print_exc()
        ok = False
    sys.exit(0 if ok else 1)
