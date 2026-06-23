# Leiratozó – Flask + Whisper + Diarizáció

Magyar nyelvű hangfájl-leiratozó webalkalmazás, amely **pyannote** beszélőazonosítást és **Whisper**-alapú átírást kombinál. Az eredmény beszélőnkénti, időbélyeges JSON és tiszta szöveges leirat.

Több felhasználó egyszerre is használhatja: a feldolgozás **sorban (queue)** zajlik, és bármely látogató valós időben láthatja az éppen zajló leiratozás előrehaladását.

## Architektúra

```
Feltöltött fájl  →  queue/ mappába kerül  →  SQLite queue tábla
                                                     │
                                               _queue_worker (háttér thread)
                                                     │
                                         ┌───────────▼────────────┐
                                         │    convert.py          │ → audio.wav
                                         │    diarization.py      │ → diarization_result.json
                                         │    transcript_after_   │ → final_transcription_*.json
                                         │    diarization.py      │   + final_text_*.txt
                                         └────────────────────────┘
```

### Pipeline részletei

```
convert.py              → audio.wav (16kHz mono WAV, ffmpeg)
      │
diarization.py          → diarization_result.json
(pyannote 3.1, GPU)       Beszélői szegmensek + időbélyegek
      │
transcript_after_diarization.py
(Trendency/whisper-large-v3-hu, GPU)
Minden turn numpy array-ként memóriában
→ manuális batch loop (ASR_BATCH_SIZE méretű csoportok)
→ PROGRESS: X/Y kimenet a frontendnek
      │
final_transcription_TIMESTAMP.json + final_text_TIMESTAMP.txt
```

**Pipeline elvek:**
- A diarizáció és az ASR **sorosan fut** – mindkettő a teljes GPU-t kapja.
- Az `audio.wav` egyszer töltődik be memóriába; nincs per-turn lemez I/O.
- Az ASR **manuális batch loop**-ban fut (`ASR_BATCH_SIZE` turn/batch), minden batch után `PROGRESS: X/Y` kerül stdout-ra.
- Sem LLM-alapú finomítás, sem Fast Whisper nem fut automatikusan a pipeline-ban.

## Feldolgozási sor (Queue)

A rendszer egyszerre csak egy leiratozást végez (GPU korlát), de **tetszőleges számú feladatot fogad el** és sorban dolgozza fel őket.

### Működés

1. Felhasználó feltölti a fájlját → `POST /upload` **azonnal** visszatér: `{queue_id, position}`
2. A fájl a `queue/` mappába kerül; a job bekerül a SQLite `queue` táblába `pending` státusszal
3. A háttér-worker (`_queue_worker`) felveszi a következő `pending` job-ot → `running` → `done`/`failed`
4. Bármely látogató lekérdezheti az aktuális állapotot a `GET /queue_status` végponton

### Sor státuszok

| Státusz | Leírás |
|---|---|
| `pending` | Várakozik a sorban |
| `running` | Jelenleg feldolgozás alatt |
| `done` | Kész, eredmény elérhető |
| `failed` | Hibával ért véget |

### Queue DB séma (`queue` tábla)

| Oszlop | Tartalom |
|---|---|
| `id` | Automatikus azonosító |
| `ip` | Feltöltő IP-je |
| `original_fn` | Eredeti fájlnév |
| `stored_fn` | Egyedi belső fájlnév (`TIMESTAMP_eredeti.ext`) |
| `queued_at` | Sorba kerülés időpontja |
| `status` | `pending` / `running` / `done` / `failed` |
| `started_at` | Feldolgozás kezdete |
| `finished_at` | Feldolgozás vége |
| `result_fn` | Kimeneti JSON fájlnév |
| `error_msg` | Hibaüzenet (ha `failed`) |

### Újraindítás utáni helyreállítás

Ha a service leáll feldolgozás közben, az app induláskor automatikusan visszaállítja a `running` státuszú job-okat `pending`-re, így azok újrafeldolgozásra kerülnek.

## Webes felület – többfelhasználós mód

```
┌─────────────────────────────────────────────────────────────────┐
│  [Sorban állsz: 2. helyen]  [Folyamatban: felvétel_hosszu.mp3] │  ← badge-ek
│                                                                  │
│  [ ✓ Konvertálás ] › [ ⟳ Diarizáció ] › [ ○ Leiratozás ]      │
│  [██████████████████░░░░░░░░░░░░░░░░░░]  38%                   │
│  Eltelt: 3:42               Becsült hátralévő: ~4:18            │
│                                                                  │
│  Feldolgozási sor:                                               │
│  ● Fut: felvétel_hosszu.mp3                                      │
│  ◆ 1. Várakozik – a te fájlod                                    │
│  ○ 2. Várakozik                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Felhasználói állapotok

| Állapot | Látható |
|---|---|
| **Idle** (semmi nem fut) | Csak feltöltő form |
| **Watcher** (más job fut, nincs saját) | Progress kártya (olvasható), "Folyamatban: …" badge |
| **Pending** (saját job vár a sorban) | Progress kártya az aktuális job haladásával + "Sorban állsz: X. helyen" badge |
| **Running** (saját job fut) | Teljes progress kártya (lépésjelző, bar, timer) |
| **Done** | Leirat szöveg + letöltés gomb |
| **Failed** | Hibaüzenet, újra lehet tölteni |

### Session tartósság

A `queue_id` a böngésző `sessionStorage`-ában tárolódik. Oldal újratöltése után a kliens automatikusan folytatja a saját job figyelését – az állapot nem vész el.

### Folyamat kijelzés részletei

| Lépés | Progress forrása | Bar tartomány |
|---|---|---|
| Konvertálás | Kliens-oldali lineáris interpoláció (~5s) | 0 → 5% |
| Diarizáció | Szerver `elapsed` + `estimatedDiarize` alapú idő-interpoláció | 5 → 22% |
| ASR | `PROGRESS: X/Y` sorok (batch-onként) | 22 → 100% |

Az eltelt idő számítása **szerver-oldali** `elapsed` értékre épül (200ms-es kliens-oldali simítással), így watchers számára is pontos – nem számít, mikor töltötte be az oldalt.

## Időbecslés

```
estimatedTotal = 5s (convert) + duration × diarize_factor + duration × asr_factor
```

- `diarize_factor` és `asr_factor` = az utolsó 15 futás **mediánja** (outlier-robusztus)
- Legalább **2 érvényes futás** szükséges a becslés megjelenítéséhez
- Az időtartam mérése **pydub**-bal történik (nem fájlméret-becsléssel)

### Statisztika tárolás (`logs` tábla)

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

Az első `python app.py` indításkor a program bekéri és `.env`-be menti a szükséges értékeket. Kézzel is szerkeszthető:

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

| ASR_BATCH_SIZE | ASR_NUM_BEAMS | Becsült VRAM | Jelleg |
|---|---|---|---|
| 6 | 5 | ~11–12 GB | **alapértelmezett** (pontos, lassabb) |
| 8 | 3 | ~10 GB | egyensúly |
| 16 | 1 | ~5–6 GB | greedy, gyors |
| 24 | 1 | ~9–10 GB | greedy, nagy batch |

**RTX 4070 Ti SUPER (16 GB):** `ASR_BATCH_SIZE=6`, `ASR_NUM_BEAMS=5` ajánlott.

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
| `/upload` | POST | Fájl feltöltése, sorba állítás – azonnali válasz |
| `/queue_status[?id=X]` | GET | Sor állapota + adott job státusza |
| `/status` | GET | Aktuálisan futó job live progress-e |
| `/stats` | GET | Futásnapló és feldolgozási statisztikák (JSON) |
| `/download/<filename>` | GET | Kész leirat letöltése |
| `/stop` | POST | Futó leiratozás leállítása |

### `POST /upload` válasz

```json
{ "queue_id": 5, "position": 1 }
```

| Mező | Leírás |
|---|---|
| `queue_id` | Egyedi azonosító, tárolni kell (`sessionStorage`) |
| `position` | Hányadik a sorban (1 = azonnal feldolgozásra kerül) |

### `GET /queue_status?id=5` válasz

```json
{
  "running": { "id": 4, "filename": "felvétel.mp3" },
  "pending_count": 1,
  "queue": [
    { "id": 5, "filename": "sajat.mp3", "queued_at": "2026-06-23T10:14:22" }
  ],
  "my_job": {
    "id": 5, "status": "pending",
    "filename": "sajat.mp3", "position": 1,
    "result_fn": null, "error_msg": null, "result_text": ""
  },
  "progress": {
    "step": 2, "stepName": "diarize", "percent": 14,
    "elapsed": 67, "estimatedTotal": 340,
    "estimatedDiarize": 62, "estimatedASR": 273,
    "remaining": 273, "detail": "", "currentStep": "Diarizáció..."
  }
}
```

Amikor `my_job.status === "done"`, a `result_text` tartalmazza a kész leiratot és a `result_fn` a letöltési nevet.

### `GET /status` válasz

```json
{
  "step": 3, "stepName": "transcribe", "percent": 52,
  "elapsed": 180, "estimatedTotal": 340,
  "estimatedDiarize": 62, "estimatedASR": 273,
  "remaining": 160, "detail": "16/31 turn",
  "currentStep": "16/31 turn kész"
}
```

### `GET /stats` válasz

```json
{
  "count": 5,
  "runs": [
    {
      "id": 5, "filename": "felvétel.mp3",
      "duration_s": 342.1, "diarize_s": 68.4,
      "asr_s": 271.3, "total_s": 344.7,
      "when": "2026-06-23 10:14:22"
    }
  ]
}
```

## Könyvtárstruktúra

```
leiratozo/
├── app.py                          # Flask szerver, queue, worker, becslési logika
├── convert.py                      # ffmpeg-alapú audio konvertálás
├── diarization.py                  # pyannote diarizáció
├── transcript_after_diarization.py # Batch ASR (Trendency, manuális batch loop)
├── fast_whisper_transcribe.py      # Standalone: faster-whisper referencia-leirat
├── llm_refine.py                   # Standalone: Ollama-alapú szövegfinomítás
├── step1_merge_diar.py             # Standalone: diarizációs szegmensek egyesítése
├── step2_split_audio.py            # Standalone: audio felszeletelés
├── step3_transcribe.py             # Standalone: ASR pipeline
├── check_gpu.py                    # GPU elérhetőség ellenőrzése
├── transcriber.service             # systemd service unit
├── Dockerfile                      # CUDA-kompatibilis konténer
├── requirements.txt                # Python függőségek
├── static/                         # Ikonok, statikus fájlok
└── templates/
    └── index.html                  # Web UI (queue, progress, statisztikák)
```

### Futás közben létrehozott mappák (gitignore-ban)

```
queue/                    # Feltöltött, feldolgozásra váró fájlok (törlődnek feldolgozás után)
uploads/                  # Régi; jelenleg nem használt
logs/                     # SQLite log (transcriber.db – logs + queue táblák)
cache/                    # HuggingFace model cache
templates/transcripts/    # Kész leiratok (30 perc után törlődnek)
audio.wav                 # Konvertált audio (felülíródik minden feldolgozásnál)
diarization_result.json   # Diarizáció kimenete (felülíródik)
```

## Standalone eszközök

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

# 3. Átírás
python step3_transcribe.py \
  --chunks runtime/chunks_manifest.json \
  --out runtime/asr_output.json \
  --word-ts fw    # off | hf | fw | approx
```

## Teljesítmény

Tipikus futásidők RTX 4070 Ti SUPER GPU-val (16 GB VRAM), `ASR_BATCH_SIZE=6`, `ASR_NUM_BEAMS=5`:

| Hanganyag | Diarizáció | ASR | Összesen |
|---|---|---|---|
| 10 perc | ~20 mp | ~1–2 perc | ~2 perc |
| 30 perc | ~1 perc | ~4–5 perc | ~5–6 perc |
| 60 perc | ~2 perc | ~8–10 perc | ~10–12 perc |

## Hibaelhárítás

**`CUDA out of memory`**
- Csökkentsd `ASR_BATCH_SIZE`-t (pl. `4`) vagy `ASR_NUM_BEAMS`-t (pl. `3` vagy `1`)
- Ha más folyamat is foglalja a VRAM-ot: `sudo systemctl restart transcriber`

**Rossz időbecslés**
- Az első 1–2 futás után javul (medián legalább 2 mérésből számít)
- Régi, hibás adatok törlése:
  ```bash
  sqlite3 logs/transcriber.db "DELETE FROM logs WHERE diarize_time = 0;"
  ```

**Job beragad `running` státuszban**
- Újraindításkor automatikusan `pending`-be kerül vissza
- Kézzel: `sqlite3 logs/transcriber.db "UPDATE queue SET status='pending' WHERE status='running';"`

**`ValueError: max_new_tokens` exceeds `max_target_positions`**
- `max_new_tokens` maximum `444` lehet (448 − 4 decoder prefix token)

**Diarizáció hibával áll le**
- Ellenőrizd a `HUGGINGFACE_TOKEN` értékét
- HuggingFace oldalon el kell fogadni a `pyannote/speaker-diarization-3.1` feltételeit

**Lassú futás CPU-n**
- Az összes modell GPU-t igényel; CPU-n 5–10× lassabb
- CUDA ellenőrzése: `python check_gpu.py`
