# Leiratozó – Flask + Whisper + Diarizáció

Magyar nyelvű hangfájl-leiratozó webalkalmazás, amely **pyannote** beszélőazonosítást és **Whisper**-alapú átírást kombinál. Az eredmény beszélőnkénti, időbélyeges JSON és tiszta szöveges leirat.

Több felhasználó egyszerre is használhatja: a feldolgozás **sorban (queue)** zajlik, és bármely látogató valós időben láthatja az éppen zajló leiratozás előrehaladását.

---

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
- Minden 25 mp-nél hosszabb turn automatikusan egyenlő részekre bomlik az ASR előtt (`split_long_turns`), majd a szöveg visszafűződik (`merge_sub_chunks`). Ez garantálja, hogy minden chunk Whisper 30 mp-es ablakán belül marad – belső sliding window nem kell, varrathiba nem léphet fel.
- Az ASR **manuális batch loop**-ban fut (`ASR_BATCH_SIZE` chunk/batch), minden batch után `PROGRESS: X/Y` kerül stdout-ra.
- Sem LLM-alapú finomítás, sem Fast Whisper nem fut automatikusan a pipeline-ban.

---

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

---

## Webes felület – többfelhasználós mód

```
┌─────────────────────────────────────────────────────────────────┐
│  [Sorban állsz: 2. helyen]  [Folyamatban: felvétel_hosszu.mp3] │
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
| **Idle** | Feltöltő form |
| **Watcher** (más job fut) | Progress kártya (olvasható), "Folyamatban: …" badge |
| **Pending** (saját job vár) | Progress kártya az aktuális jobbal + "Sorban állsz: X. helyen" badge |
| **Running** (saját job fut) | Teljes progress kártya (lépésjelző, bar, timer) |
| **Done** | Leirat szöveg + letöltés gomb |
| **Failed** | Hibaüzenet |

### Session tartósság

A `queue_id` a böngésző `sessionStorage`-ában tárolódik. Oldal újratöltése után a kliens automatikusan folytatja a saját job figyelését.

### Folyamat kijelzés

| Lépés | Progress forrása | Bar tartomány |
|---|---|---|
| Konvertálás | Kliens-oldali lineáris interpoláció (~5s) | 0 → 5% |
| Diarizáció | Szerver `elapsed` + `estimatedDiarize` idő-interpoláció | 5 → 22% |
| ASR | `PROGRESS: X/Y` sorok (batch-onként) | 22 → 100% |

Az eltelt idő számítása szerver-oldali `elapsed` értékre épül (200ms-es kliens-oldali simítással), így watchers számára is pontos.

---

## Időbecslés

```
estimatedTotal = 5s (convert) + duration × diarize_factor + duration × asr_factor
```

- `diarize_factor` és `asr_factor` = az utolsó 15 futás **mediánja** (outlier-robusztus)
- Legalább **2 érvényes futás** szükséges a becslés megjelenítéséhez
- Az időtartam mérése **pydub**-bal történik

### Statisztika tárolás (`logs` tábla)

| Oszlop | Tartalom |
|---|---|
| `filename` | feltöltött fájl neve |
| `filetype` | kiterjesztés |
| `duration` | hanganyag hossza másodpercben |
| `diarize_time` | diarizáció futásideje |
| `asr_time` | ASR futásideje |
| `runtime` | teljes feldolgozási idő |
| `result_fn` | kimeneti fájl neve (letöltéshez) |
| `start_time` / `end_time` | időbélyegek |

### Kész leiratok letöltése a statisztika táblából

A statisztika panelben minden sorban megjelenik egy **TXT** és **JSON** letöltési link, amíg a fájl elérhető. A leiratok **30 perccel a létrehozásuk után törlődnek** (a következő job befejezésekor fut a cleanup).

---

## Modellválasztás és finomhangolás

Az aktív ASR modell a `.env` `WHISPER_MODEL_DIR` változójával állítható:

| Állapot | `WHISPER_MODEL_DIR` értéke |
|---|---|
| Eredeti (alap) modell | `""` (üres) |
| Finomhangolt modell | `/srv/transcriber_app/finetune_output` |

A váltás és az **alkalmazás újraindítása** a `/finetune` aloldalon keresztül végezhető – terminálhoz való hozzáférés nélkül. Az újraindítás `os.execv()` alapú: a Python folyamat saját magát cseréli le, a `.env` újratöltődik, a systemd service folyamatosan fut.

### Modell finomhangolása (lokális, privát funkció)

> **Megjegyzés:** A finomhangoláshoz szükséges forrásfájlok (`finetune_run.py`, `templates/finetune.html`) nem részei a publikus repónak – ezek a szerveren lokálisan léteznek.

A `/finetune` aloldalon:
- Hanganyag + javított leirat párok tölthetők fel (max. 30 mp, WAV-ra konvertálva)
- A betanítás **kézzel indítható** a weboldalon – automatikus ütemezés alapból ki van kapcsolva
- Minden futás **az összes feltöltött mintán** végigmegy (nem csak az újakon), így a modell nem felejti el a korábbi anyagokat
- LoRA módszerrel tanít (rank=8, alpha=32, target: `q_proj` + `v_proj`), majd `merge_and_unload()` után teljes modellként menti

#### Tanítóadatok tárolási helye

| Adat | Elérési út |
|---|---|
| Hanganyagok | `/srv/transcriber_app/training_data/` (WAV, 16kHz mono) |
| Leiratok | `/srv/transcriber_app/logs/transcriber.db` → `training_data` tábla |
| Tanított modell | `/srv/transcriber_app/finetune_output/` (~2.9 GB) |

#### `training_data` DB tábla

| Oszlop | Tartalom |
|---|---|
| `id` | Automatikus azonosító |
| `audio_fn` | Hangfájl neve a `training_data/` mappában |
| `transcript` | Javított leirat szövege |
| `uploaded_at` | Feltöltés időpontja |
| `duration_s` | Hanganyag hossza másodpercben |
| `used_in_run` | Melyik futásban lett először felhasználva (0 = még nem) |

#### Automatikus ütemezés (opcionális)

Alapból **ki van kapcsolva**. Bekapcsoláshoz `.env`-be:
```
FINETUNE_AUTO=1
FINETUNE_START_HOUR=22
FINETUNE_END_HOUR=6
```
Ha `FINETUNE_AUTO=1`, a háttér-thread 10 percenként ellenőrzi az időablakot, és ha a leiratozási sor üres, automatikusan elindítja a tanítást.

---

## Követelmények

- Python 3.11+
- NVIDIA GPU, CUDA 11.8+ (ajánlott; CPU-n is fut, de ~10× lassabb)
- `ffmpeg` a PATH-ban
- HuggingFace fiók + elfogadott feltételek a `pyannote/speaker-diarization-3.1` modellhez

---

## Telepítés

```bash
git clone https://github.com/kristivok/leiratozo.git
cd leiratozo

python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

# PyTorch CUDA 11.8
pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

pip install faster-whisper
pip install -r requirements.txt
```

---

## Konfiguráció

Az első `python app.py` indításkor a program bekéri és `.env`-be menti a szükséges értékeket.

### Alapbeállítások

| Változó | Leírás | Alapértelmezés |
|---|---|---|
| `HUGGINGFACE_TOKEN` | HF API token (kötelező diarizációhoz) | – |
| `PORT` | Flask szerver port | `58515` |
| `HF_CACHE_DIR` | Modellek cache könyvtára | `./cache` |
| `WHISPER_MODEL_ID` | HuggingFace model ID | `Trendency/whisper-large-v3-hu` |
| `WHISPER_MODEL_DIR` | Lokális model mappa (felülírja a model ID-t) | – |
| `DIARIZATION_DEVICE` | Diarizáció eszköze (`cuda`/`cpu`) | `cuda` |
| `DIARIZATION_BATCH_SIZE` | Pyannote batch méret | `8` |
| `ASR_BATCH_SIZE` | Hány turn kerül egy GPU batch-be | `6` |
| `ASR_NUM_BEAMS` | Beam search szélessége (1=greedy, 5=pontosabb) | `5` |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA memória-allokátor | `expandable_segments:True` |

### Finomhangolás beállításai

| Változó | Leírás | Alapértelmezés |
|---|---|---|
| `FINETUNE_AUTO` | Automatikus ütemezés (`0`=ki, `1`=be) | `0` |
| `FINETUNE_START_HOUR` | Auto ablak kezdete (óra) | `22` |
| `FINETUNE_END_HOUR` | Auto ablak vége (óra) | `6` |
| `FINETUNE_EPOCHS` | Tanítási epoch-ok száma | `3` |
| `FINETUNE_BATCH_SIZE` | Batch méret tanítás közben | `4` |
| `FINETUNE_LR` | Tanulási ráta | `1e-4` |
| `FINETUNE_LORA_RANK` | LoRA rang | `8` |
| `FINETUNE_LORA_ALPHA` | LoRA alpha | `32` |
| `FINETUNE_MAX_STEPS` | Max lépések (`0`=korlátlan) | `0` |

### VRAM és pontosság

| ASR_BATCH_SIZE | ASR_NUM_BEAMS | Becsült VRAM | Jelleg |
|---|---|---|---|
| 6 | 5 | ~11–12 GB | **alapértelmezett** |
| 8 | 3 | ~10 GB | egyensúly |
| 16 | 1 | ~5–6 GB | greedy, gyors |

---

## Futtatás

```bash
source venv/bin/activate
python app.py
```

Megnyitás: `http://localhost:58515`

### Systemd szolgáltatásként

```bash
sudo cp transcriber.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now transcriber
journalctl -u transcriber -f
```

---

## API végpontok

| Végpont | Metódus | Leírás |
|---|---|---|
| `/` | GET | Web UI |
| `/upload` | POST | Fájl feltöltése, sorba állítás |
| `/queue_status[?id=X]` | GET | Sor állapota + adott job státusza |
| `/status` | GET | Aktuálisan futó job live progress-e |
| `/stats` | GET | Futásnapló és statisztikák (JSON) |
| `/download/<filename>` | GET | Kész leirat letöltése |
| `/stop` | POST | Futó leiratozás leállítása |
| `/restart` | POST | Alkalmazás újraindítása (modellváltás után) |
| `/finetune` | GET | Finomhangolás aloldal (lokális) |
| `/finetune/upload` | POST | Tanítóadat feltöltése |
| `/finetune/data` | GET | Feltöltött minták listája (JSON) |
| `/finetune/data/<id>/delete` | POST | Minta törlése |
| `/finetune/audio/<filename>` | GET | Hanganyag letöltése |
| `/finetune/status` | GET | Finomhangolás státusz, log, futási előzmények |
| `/finetune/trigger` | POST | Finomhangolás kézi indítása |
| `/finetune/stop` | POST | Futó finomhangolás leállítása |
| `/finetune/activate` | POST | Finomhangolt modell aktiválása (.env) |
| `/finetune/revert` | POST | Visszaállítás az eredeti modellre (.env) |

### `POST /upload` válasz

```json
{ "queue_id": 5, "position": 1 }
```

### `GET /queue_status?id=5` válasz

```json
{
  "running": { "id": 4, "filename": "felvétel.mp3" },
  "pending_count": 1,
  "queue": [{ "id": 5, "filename": "sajat.mp3", "queued_at": "..." }],
  "my_job": {
    "id": 5, "status": "pending", "filename": "sajat.mp3",
    "position": 1, "result_fn": null, "result_text": ""
  },
  "progress": {
    "step": 2, "stepName": "diarize", "percent": 14,
    "elapsed": 67, "estimatedTotal": 340,
    "estimatedDiarize": 62, "estimatedASR": 273,
    "remaining": 273, "detail": "", "currentStep": "Diarizáció..."
  }
}
```

### `GET /finetune/status` válasz

```json
{
  "running": false,
  "queue_busy": false,
  "total_samples": 12,
  "model_ready": true,
  "model_dir": "/srv/transcriber_app/finetune_output",
  "active_model": "finetune",
  "active_label": "/srv/transcriber_app/finetune_output",
  "log_tail": ["STEP: 45/60 loss=0.3241", "EPOCH: 3/3 avg_loss=0.3189", "=== Kész ==="],
  "runs": [{ "id": 2, "started_at": "...", "finished_at": "...", "status": "done",
             "samples_used": 12, "last_loss": 0.3189, "error_msg": null }]
}
```

---

## Könyvtárstruktúra

```
leiratozo/
├── app.py                          # Flask szerver, queue, worker, becslés, modellváltás, finomhangolás API
├── convert.py                      # ffmpeg-alapú audio konvertálás
├── diarization.py                  # pyannote diarizáció
├── transcript_after_diarization.py # Batch ASR (manuális batch loop)
├── fast_whisper_transcribe.py      # Standalone: faster-whisper referencia
├── llm_refine.py                   # Standalone: Ollama szövegfinomítás
├── step1_merge_diar.py             # Standalone: diarizációs szegmensek egyesítése
├── step2_split_audio.py            # Standalone: audio felszeletelés
├── step3_transcribe.py             # Standalone: ASR pipeline
├── check_gpu.py                    # GPU ellenőrzés
├── transcriber.service             # systemd service unit
├── requirements.txt                # Python függőségek
├── static/                         # Ikonok, statikus fájlok
└── templates/
    └── index.html                  # Web UI (queue, progress, statisztikák)
```

> A finomhangoláshoz tartozó fájlok (`finetune_run.py`, `templates/finetune.html` stb.) **nem részei a publikus repónak** – csak a szerveren érhetők el lokálisan.

### Futás közben létrehozott mappák (gitignore-ban)

```
queue/                    # Feltöltött, feldolgozásra váró fájlok
logs/                     # SQLite (transcriber.db: logs + queue + training_data + finetune_runs)
cache/                    # HuggingFace model cache
templates/transcripts/    # Kész leiratok (30 perc után törlődnek)
training_data/            # Finomhangoláshoz feltöltött hanganyagok
finetune_output/          # Tanított modell (~2.9 GB)
audio.wav                 # Konvertált audio (felülíródik)
diarization_result.json   # Diarizáció kimenete (felülíródik)
```

---

## Hibaelhárítás

**`CUDA out of memory`**
```bash
# .env-ben csökkentsd:
ASR_BATCH_SIZE=4
ASR_NUM_BEAMS=1
```

**Rossz időbecslés**
```bash
sqlite3 logs/transcriber.db "DELETE FROM logs WHERE diarize_time = 0;"
```

**Job beragad `running` státuszban**
```bash
sqlite3 logs/transcriber.db "UPDATE queue SET status='pending' WHERE status='running';"
```

**`ValueError: max_new_tokens` exceeds `max_target_positions`**
- `max_new_tokens` maximum `444` lehet

**Diarizáció hibával áll le**
- Ellenőrizd a `HUGGINGFACE_TOKEN` értékét
- HuggingFace oldalon el kell fogadni a `pyannote/speaker-diarization-3.1` feltételeit

**Hosszú turn-ök – hiányzó szöveg**
- Whisper maximális ablaka 30 mp. Ha egy diarizációs turn ennél hosszabb volt és a pipeline belső sliding window-ja varrathiba miatt kihagyott tartalmat, a `split_long_turns()` (25 mp-es határ) megoldja: ez automatikusan fut minden leiratozásnál.

**Finomhangolás: `No module named 'peft'`**
```bash
/srv/transcriber_app/venv/bin/pip install "peft>=0.11.0" "datasets>=2.20.0"
```

**Finomhangolás: `input_ids` hiba Whisper-rel**
- A `LoraConfig`-ban **nem szabad** `task_type=TaskType.SEQ_2_SEQ_LM` paramétert megadni – Whisper `input_features`-t vár, nem `input_ids`-t.
