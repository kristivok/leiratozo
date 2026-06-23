# Leiratozó – Flask + Whisper + Diarizáció

Magyar nyelvű hangfájl-leiratozó webalkalmazás, amely **pyannote** beszélőazonosítást és **Whisper**-alapú átírást kombinál. Az eredmény beszélőnkénti, időbélyeges JSON és tiszta szöveges leirat.

## Architektúra

```
Feltöltött fájl
      │
      ▼
 convert.py              → audio.wav (16kHz mono WAV, ffmpeg)
      │
      ▼
 diarization.py          → diarization_result.json
 (pyannote 3.1, GPU)       Beszélői szegmensek + időbélyegek
      │
      ▼
 transcript_after_diarization.py
 (Trendency/whisper-large-v3-hu, GPU)
 Minden turn numpy array-ként memóriában
 → manuális batch loop (ASR_BATCH_SIZE méretű csoportok)
 → PROGRESS: X/Y kimenet a frontendnek
      │
      ▼
 final_transcription_TIMESTAMP.json + final_text_TIMESTAMP.txt
```

**Pipeline elvek:**
- A diarizáció és az ASR **sorosan fut** – mindkettő a teljes GPU-t kapja.
- Az `audio.wav` egyszer töltődik be memóriába; nincs per-turn lemez I/O.
- Az ASR **manuális batch loop**-ban fut (`ASR_BATCH_SIZE` turn/batch), minden batch után `PROGRESS: X/Y` kerül stdout-ra.
- Sem LLM-alapú finomítás, sem Fast Whisper nem fut automatikusan a pipeline-ban.

## Webes felület

A feltöltés gomb megnyomásától kezdve folyamatos visszajelzés:

```
[ ✓ Konvertálás ] › [ ⟳ Diarizáció ] › [ ○ Leiratozás ]

[████████████████████░░░░░░░░░░░░░░░░]  52%

ASR: 16/31 turn                Eltelt: 3:42
              Becsült hátralévő: ~3:18

  Turn 16/31 kész
```

- **Step indicator**: 3 lépés (kész ✓ / aktív pulzáló / várakozó)
- **Progress bar**: folyamatos animáció – konvertálásnál lineáris, diarizációnál idő-alapú interpoláció, ASR-nél batch-pontos
- **Eltelt idő**: kliens-oldali számláló (gombnyomástól, 200ms frissítés)
- **Becsült hátralévő**: `estimatedTotal − eltelt`, automatikusan csökken
- **Statisztikák gomb**: futásnapló táblázat + medián feldolgozási sebesség

### Folyamat kijelzés részletei

| Lépés | Progress forrása | Bar tartomány |
|---|---|---|
| Konvertálás | lineáris idő-interpoláció (~5s) | 0 → 5% |
| Diarizáció | `estimatedDiarize` alapú idő-interpoláció | 5 → 22% |
| ASR | `PROGRESS: X/Y` sorok (batch-onként) | 22 → 100% |

## Időbecslés

A becsült feldolgozási idő **medián-alapú, per-lépéses** számítással működik:

```
estimatedTotal = 5s (convert) + duration × diarize_factor + duration × asr_factor
```

- `diarize_factor` és `asr_factor` = az utolsó 15 futás **mediánja** (outlier-robusztus)
- Legalább **2 érvényes futás** szükséges a becslés megjelenítéséhez
- Az időtartam mérése **pydub**-bal történik (nem fájlméret-becsléssel)

### Statisztika tárolás

Minden leiratozás adatai az `logs/transcriber.db` SQLite adatbázisba kerülnek:

| Oszlop | Tartalom |
|---|---|
| `filename` | feltöltött fájl neve |
| `filetype` | kiterjesztés |
| `duration` | hanganyag hossza másodpercben (pydub mérés) |
| `diarize_time` | diarizáció futásideje másodpercben |
| `asr_time` | ASR futásideje másodpercben |
| `runtime` | teljes feldolgozási idő |
| `start_time` / `end_time` | időbélyegek |

A statisztikák a webes felületen a **Statisztikák** gombbal tekinthetők meg.

## Követelmények

- Python 3.11+
- NVIDIA GPU, CUDA 11.8+ (ajánlott; CPU-n is fut, de ~10× lassabb)
- `ffmpeg` a PATH-ban
- HuggingFace fiók + elfogadott feltételek a `pyannote/speaker-diarization-3.1` modellhez

## Telepítés

```bash
git clone https://github.com/kristivok/leiratozo.git
cd leiratozo

python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip setuptools wheel

# PyTorch CUDA 11.8 – GPU-s futtatáshoz
pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# CPU-only PyTorch (ha nincs GPU)
# pip install torch==2.6.0 torchaudio==2.6.0

pip install faster-whisper
pip install -r requirements.txt
```

## Konfiguráció

Az első `python app.py` indításkor a program bekéri és `.env`-be menti a szükséges értékeket. Kézzel is szerkeszthető a `.env` fájl:

| Változó | Leírás | Alapértelmezés |
|---|---|---|
| `HUGGINGFACE_TOKEN` | HF API token (kötelező diarizációhoz) | – |
| `PORT` | Flask szerver port | `58515` |
| `HF_CACHE_DIR` | Modellek cache könyvtára | `./cache` |
| `WHISPER_MODEL_ID` | HuggingFace model ID | `Trendency/whisper-large-v3-hu` |
| `WHISPER_MODEL_DIR` | Lokális model mappa (opcionális, felülírja a model ID-t) | – |
| `DIARIZATION_DEVICE` | Diarizáció eszköze (`cuda`/`cpu`) | `cuda` |
| `DIARIZATION_BATCH_SIZE` | Pyannote batch méret | `8` |
| `ASR_BATCH_SIZE` | Hány turn kerül egy GPU batch-be | `6` |
| `ASR_NUM_BEAMS` | Beam search szélessége (1=greedy/gyors, 5=pontosabb) | `5` |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA memória-allokátor | `expandable_segments:True` |

### VRAM és pontosság hangolása

Az `ASR_BATCH_SIZE` és `ASR_NUM_BEAMS` együtt határozza meg a VRAM-fogyasztást:

| ASR_BATCH_SIZE | ASR_NUM_BEAMS | Becsült VRAM | Jelleg |
|---|---|---|---|
| 6 | 5 | ~11–12 GB | **alapértelmezett** (pontos, lassabb) |
| 8 | 3 | ~10 GB | egyensúly |
| 16 | 1 | ~5–6 GB | greedy, gyors |
| 24 | 1 | ~9–10 GB | greedy, nagy batch |

**RTX 4070 Ti SUPER (16 GB):** `ASR_BATCH_SIZE=6`, `ASR_NUM_BEAMS=5` ajánlott.

### Másik ASR modell használata

```bash
# .env-ben:
WHISPER_MODEL_ID=openai/whisper-large-v3
```

## Futtatás

### Fejlesztői módban

```bash
source venv/bin/activate
python app.py
```

Megnyitás: `http://localhost:58515`

### Systemd szolgáltatásként (Linux)

```bash
sudo cp transcriber.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now transcriber
journalctl -u transcriber -f
```

A `.env` értékeit a service automatikusan betölti (`EnvironmentFile` direktíva).

### Docker (GPU)

```bash
docker build -t leiratozo .
docker run --gpus all -p 58515:58515 \
  -e HUGGINGFACE_TOKEN=hf_... \
  leiratozo
```

## API végpontok

| Végpont | Metódus | Leírás |
|---|---|---|
| `/` | GET | Web UI |
| `/upload` | POST | Fájl feltöltése és leiratozás indítása |
| `/status` | GET | Folyamat állapota – step, percent, elapsed, remaining, detail |
| `/stats` | GET | Futásnapló és feldolgozási statisztikák (JSON) |
| `/download/<filename>` | GET | Kész leirat letöltése |
| `/stop` | POST | Futó leiratozás leállítása |

### `/status` válasz mezők

```json
{
  "step": 2,
  "stepName": "diarize",
  "percent": 14,
  "elapsed": 67,
  "estimatedTotal": 340,
  "estimatedDiarize": 62,
  "estimatedASR": 273,
  "remaining": 273,
  "detail": "",
  "currentStep": "Diarizáció kész"
}
```

### `/stats` válasz mezők

```json
{
  "count": 5,
  "runs": [
    {
      "id": 5,
      "filename": "felvétel.mp3",
      "duration_s": 342.1,
      "diarize_s": 68.4,
      "asr_s": 271.3,
      "total_s": 344.7,
      "when": "2026-06-23 10:14:22"
    }
  ]
}
```

## Könyvtárstruktúra

```
leiratozo/
├── app.py                          # Flask szerver, pipeline vezérlés, becslési logika
├── convert.py                      # ffmpeg-alapú audio konvertálás
├── diarization.py                  # pyannote diarizáció
├── transcript_after_diarization.py # Batch ASR (Trendency, manuális batch loop)
├── fast_whisper_transcribe.py      # Standalone: faster-whisper referencia-leirat
├── llm_refine.py                   # Standalone: Ollama-alapú szövegfinomítás
├── step1_merge_diar.py             # Standalone: diarizációs szegmensek egyesítése
├── step2_split_audio.py            # Standalone: audio felszeletelés
├── step3_transcribe.py             # Standalone: ASR pipeline (szó-szintű TS opcióval)
├── check_gpu.py                    # GPU elérhetőség ellenőrzése
├── finetune_prepare.py             # Finomhangolási adat-előkészítő
├── finetune_chunks.py              # Finomhangolási chunkolás
├── finetune_transcript.py          # Finomhangolási átírás
├── transcriber.service             # systemd service unit
├── Dockerfile                      # CUDA-kompatibilis konténer
├── requirements.txt                # Python függőségek
├── static/                         # Ikonok, statikus fájlok
└── templates/
    └── index.html                  # Web UI (progress, statisztikák)
```

### Futás közben létrehozott mappák (gitignore-ban)

```
uploads/                  # Feltöltött fájlok (átmeneti)
cache/                    # HuggingFace model cache
logs/                     # SQLite log (transcriber.db)
templates/transcripts/    # Kész leiratok (30 perc után törlődnek)
audio.wav                 # Konvertált audio (felülíródik minden kérésnél)
diarization_result.json   # Diarizáció kimenete (felülíródik)
```

## Standalone eszközök

A `step1/2/3` scriptek önállóan is futtathatók, pl. batch feldolgozáshoz:

```bash
# 1. Diarizációs szegmensek összevonása
python step1_merge_diar.py \
  --in diarization_result.json \
  --out runtime/merged_segments.json \
  --max_gap 2.0

# 2. Audio felszeletelés
python step2_split_audio.py \
  --audio audio.wav \
  --segments runtime/merged_segments.json \
  --out runtime/chunks_manifest.json \
  --chunk_dir chunks/

# 3. Átírás (szó-szintű időbélyeg opciókkal)
python step3_transcribe.py \
  --chunks runtime/chunks_manifest.json \
  --out runtime/asr_output.json \
  --word-ts fw    # off | hf | fw | approx
```

### Fast Whisper – standalone referencia-leirat

Nem fut automatikusan (GPU-versengés és HU minőségi tradeoff miatt). Manuálisan:

```bash
python fast_whisper_transcribe.py
# Kimenet: templates/transcripts/fast_transcript.json
```

### LLM finomítás – standalone (Ollama)

Nem fut automatikusan a pipeline-ban. Manuálisan:

```bash
python llm_refine.py \
  templates/transcripts/final_transcription_TIMESTAMP.json \
  output_refined.txt
# Alapértelmezett modell: llama3.1:70b
# Felülírható: OLLAMA_MODEL=llama3.1:8b python llm_refine.py ...
```

## Teljesítmény

Tipikus futásidők RTX 4070 Ti SUPER GPU-val (16 GB VRAM), `ASR_BATCH_SIZE=6`, `ASR_NUM_BEAMS=5`:

| Hanganyag | Diarizáció | ASR | Összesen |
|---|---|---|---|
| 10 perc | ~20 mp | ~1–2 perc | ~2 perc |
| 30 perc | ~1 perc | ~4–5 perc | ~5–6 perc |
| 60 perc | ~2 perc | ~8–10 perc | ~10–12 perc |

A pontos adatok a **Statisztikák** gombban láthatók a webes felületen.

## Hibaelhárítás

**`ModuleNotFoundError: No module named 'faster_whisper'`**
```bash
venv/bin/pip install faster-whisper
```

**GPU nem látszik (`nvidia-smi` sem mutat Python folyamatot)**
- `.env`-ben ellenőrizd: `DIARIZATION_DEVICE=cuda`
- `python -c "import torch; print(torch.cuda.is_available())"`

**`CUDA out of memory`**
- Csökkentsd `ASR_BATCH_SIZE`-t (pl. `4`) vagy `ASR_NUM_BEAMS`-t (pl. `3` vagy `1`)
- Ha más folyamat is foglalja a VRAM-ot: `sudo systemctl restart transcriber`

**Rossz időbecslés (nagyon eltér a valóságtól)**
- Az első 1–2 futás után javul (medián legalább 2 mérésből számít)
- A régi, hibás adatok kiszűrésére a `logs/transcriber.db`-ből töröld a rossz sorokat:
  ```bash
  sqlite3 logs/transcriber.db "DELETE FROM logs WHERE diarize_time = 0;"
  ```

**Diarizáció hibával áll le**
- Ellenőrizd a `HUGGINGFACE_TOKEN` értékét
- HuggingFace oldalon el kell fogadni a `pyannote/speaker-diarization-3.1` feltételeit

**`ValueError: max_new_tokens` exceeds `max_target_positions`**
- A `GENERATE_KWARGS`-ban `max_new_tokens` maximum `444` lehet (448 − 4 decoder prefix token)

**"Már fut egy másik leiratozás!" – holott nem fut semmi**
- `rm transcriber.lock`
- Vagy a webes `/stop` gombbal állítsd le

**Lassú futás CPU-n**
- Az összes modell (pyannote, Trendency) GPU-t igényel; CPU-n 5–10× lassabb
- CUDA telepítés ellenőrzése: `python check_gpu.py`
