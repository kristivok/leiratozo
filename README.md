# Leiratozó – Flask + Whisper + Diarizáció

Magyar nyelvű hangfájl-leiratozó webalkalmazás, amely **pyannote** beszélőazonosítást és **Whisper**-alapú átírást kombinál. Az eredmény beszélőnkénti, időbélyeges JSON és tiszta szöveges leirat.

## Architektúra

```
Feltöltött fájl
      │
      ▼
 convert.py          → audio.wav (16kHz mono WAV, ffmpeg)
      │
      ├─────────────────────────────────────┐
      ▼                                     ▼
fast_whisper_transcribe.py           diarization.py
(faster-whisper large-v3, GPU)       (pyannote 3.1, GPU)
Teljes referencia-leirat              Beszélői szegmensek
fast_transcript.json                  diarization_result.json
      │                                     │
      └──────────────┬──────────────────────┘
                     ▼
        transcript_after_diarization.py
        (Trendency/whisper-large-v3-hu, turnönként, GPU)
        → Végleges leirat + szöveg mentése
```

**Párhuzamos futás:** a Fast Whisper és a diarizáció egyszerre indul. A Fast Whisper kimenetét (`fast_transcript.json`) a végleges leirat fuzzy egyeztetéssel finomítja.

## Követelmények

- Python 3.11+
- NVIDIA GPU CUDA 11.8+ (ajánlott; CPU-n is fut, de ~10× lassabb)
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
| `WHISPER_MODEL_DIR` | Lokális model mappa (opcionális) | – |
| `DIARIZATION_DEVICE` | Diarizáció eszköze (`cuda`/`cpu`) | `cuda` |
| `DIARIZATION_BATCH_SIZE` | Pyannote batch méret | `8` |

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
| `/status` | GET | Aktuális lépés szövege (polling) |
| `/download/<filename>` | GET | Kész leirat letöltése |
| `/stop` | POST | Futó leiratozás leállítása |

## Könyvtárstruktúra

```
leiratozo/
├── app.py                          # Flask szerver, pipeline vezérlés
├── convert.py                      # ffmpeg-alapú audio konvertálás
├── diarization.py                  # pyannote diarizáció
├── fast_whisper_transcribe.py      # Gyors referencia-leirat (faster-whisper)
├── transcript_after_diarization.py # Végleges leirat (Trendency, turnönként)
├── llm_refine.py                   # Opcionális Ollama-alapú szövegfinomítás
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
    └── index.html                  # Web UI
```

### Futás közben létrehozott mappák (gitignore-ban)

```
uploads/                  # Feltöltött fájlok (átmeneti)
chunks/                   # Audio szegmensek (átmeneti)
cache/                    # HuggingFace model cache
logs/                     # SQLite log (futásidők, statisztikák)
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

## Opcionális LLM finomítás (Ollama)

Ha telepítve van [Ollama](https://ollama.com/):

```bash
python llm_refine.py \
  templates/transcripts/final_transcription_TIMESTAMP.json \
  output_refined.txt
```

Alapértelmezett modell: `llama3.1:70b` (felülírható: `OLLAMA_MODEL=llama3.1:8b python llm_refine.py ...`).

## Teljesítmény

Tipikus futásidők RTX 4070 Ti SUPER GPU-val (16 GB VRAM):

| Hanganyag | Diarizáció | Fast Whisper | Trendency ASR | Összesen |
|---|---|---|---|---|
| 30 perc | ~3 perc | ~1 perc | ~5 perc | ~6 perc |
| 60 perc | ~6 perc | ~2 perc | ~10 perc | ~12 perc |

A diarizáció és a Fast Whisper párhuzamosan fut → összidő ≈ `max(diar, fw) + trendency`.

## Hibaelhárítás

**`ModuleNotFoundError: No module named 'faster_whisper'`**
```bash
venv/bin/pip install faster-whisper
```

**GPU nem látszik (`nvidia-smi` sem mutat Python folyamatot)**
- `.env`-ben ellenőrizd: `DIARIZATION_DEVICE=cuda`
- `python -c "import torch; print(torch.cuda.is_available())"`

**Diarizáció hibával áll le**
- Ellenőrizd a `HUGGINGFACE_TOKEN` értékét
- HuggingFace oldalon el kell fogadni a `pyannote/speaker-diarization-3.1` feltételeit

**"Már fut egy másik leiratozás!" – holott nem fut semmi**
- Maradt lock fájl: `rm transcriber.lock`
- Vagy a webes `/stop` gombbal állítsd le

**Lassú futás CPU-n**
- Az összes modell (Fast Whisper, pyannote, Trendency) GPU-t vár; CPU-n 5-10× lassabb
- CUDA telepítés ellenőrzése: `check_gpu.py`
