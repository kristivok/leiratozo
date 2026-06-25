# Leiratozó – Flask + Whisper + Diarizáció

Magyar nyelvű hangfájl-leiratozó webalkalmazás, amely **pyannote** beszélőazonosítást és **Whisper**-alapú átírást kombinál. Az eredmény beszélőnkénti, időbélyeges JSON és tiszta szöveges leirat.

Több felhasználó egyszerre is használhatja: a feldolgozás **sorban (queue)** zajlik, és bármely látogató valós időben láthatja az éppen zajló leiratozás előrehaladását.

---

## Architektúra

```
Feltöltött fájl
      │
      ▼  POST /upload
queue/ mappa  +  SQLite queue tábla (pending)
      │
      ▼  _queue_worker (háttér thread)
┌─────────────────────────────────────────┐
│  convert.py          → audio.wav        │
│  diarization.py      → diarization_     │
│                         result.json     │
│  transcript_after_   → final_           │
│  diarization.py         transcription_  │
│                         *.json + *.txt  │
└─────────────────────────────────────────┘
      │
  SQLite queue tábla (done/failed)
  SQLite logs tábla (statisztika)
```

---

## Pipeline részletei

### 1. Konvertálás (`convert.py`)

```
Bármilyen hangfájl → ffmpeg → audio.wav (16kHz, mono, PCM)
```

### 2. Diarizáció (`diarization.py`)

```
audio.wav → pyannote/speaker-diarization-3.1 (GPU) → diarization_result.json
```

Kimenet: `[{speaker, start, end}, ...]` – minden turn külön sorban, időbélyeggel.

### 3. Leiratozás (`transcript_after_diarization.py`)

```
diarization_result.json
      │
      ▼  normalize_and_sort_segments()
         Szűri a <50ms elemeket, rendezi start szerint
      │
      ▼  merge_consecutive_same_speaker(max_gap=1.0s)
         Azonos szomszédos speaker turn-öket összefűz,
         ha a köztük lévő szünet ≤ 1 mp
      │
      ▼  split_long_turns(max_dur=25s)
         A >25s turn-öket egyenlő részekre osztja.
         Whisper 30s ablakán belül maradnak → nincs belső
         sliding window, nincs varrathiba.
      │
      ▼  transcribe_turns()  [batch ASR]
         audio.wav egyszer betöltve memóriába
         Minden chunk → numpy array (16kHz float32)
         GPU batch: ASR_BATCH_SIZE chunk egyszerre
         GENERATE_KWARGS: language=hu, num_beams, max_new_tokens=444
         → PROGRESS: X/Y stdout-ra (batch-onként)
      │
      ▼  merge_sub_chunks()
         A split_long_turns által feldarabolt részek
         szövegét visszafűzi az eredeti turn-höz
      │
final_transcription_TIMESTAMP.json
final_text_TIMESTAMP.txt
```

**Kulcsdöntések:**
- A diarizáció és az ASR **sorosan fut** – mindkettő a teljes GPU-t kapja.
- Az `audio.wav` egyszer töltődik be memóriába; nincs per-turn lemez I/O.
- `max_new_tokens=444` – Whisper belső korlátja (`max_target_positions=448`, 4 decoder prompt token foglalt).
- A pipeline-ban nincs `chunk_length_s` – a `split_long_turns` garantálja, hogy minden chunk ≤ 25s.

---

## Feldolgozási sor (Queue)

A rendszer egyszerre csak egy leiratozást végez (GPU korlát), de **tetszőleges számú feladatot fogad el** és sorban dolgozza fel őket.

### Működés

1. `POST /upload` → fájl a `queue/` mappába kerül, job bekerül a `queue` táblába `pending` státusszal, azonnal visszatér: `{queue_id, position}`
2. `_queue_worker` (háttér thread) felveszi a következő `pending` job-ot → `running` → `done`/`failed`
3. Bármely látogató lekérdezheti az állapotot: `GET /queue_status`

### Sor státuszok

| Státusz | Leírás |
|---|---|
| `pending` | Várakozik a sorban |
| `running` | Jelenleg feldolgozás alatt |
| `done` | Kész, eredmény elérhető |
| `failed` | Hibával ért véget |

### `queue` DB tábla

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

Induláskor az app automatikusan visszaállítja a `running` státuszú job-okat `pending`-re, így leállás után azok újrafeldolgozásra kerülnek.

---

## Webes felület – többfelhasználós mód

```
┌──────────────────────────────────────────────────────────────────┐
│  [Sorban állsz: 2. helyen]  [Folyamatban: felvétel_hosszu.mp3]  │
│                                                                   │
│  [ ✓ Konvertálás ] › [ ⟳ Diarizáció ] › [ ○ Leiratozás ]       │
│  [██████████████████░░░░░░░░░░░░░░░░░░]  38%                    │
│  Eltelt: 3:42               Becsült hátralévő: ~4:18             │
│                                                                   │
│  Feldolgozási sor:                                                │
│  ● Fut: felvétel_hosszu.mp3                                       │
│  ◆ 1. Várakozik – a te fájlod                                     │
│  ○ 2. Várakozik                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Felhasználói állapotok

| Állapot | Látható |
|---|---|
| **Idle** | Feltöltő form |
| **Watcher** (más job fut) | Progress kártya (olvasható), „Folyamatban: …" badge |
| **Pending** (saját job vár) | Progress kártya az aktuális jobbal + „Sorban állsz: X. helyen" badge |
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

Az eltelt idő számítása szerver-oldali `elapsed` értékre épül (200ms-es kliens-oldali simítással) – watchers számára is pontos.

---

## Időbecslés

```
estimatedTotal = 5s (convert)
               + duration × diarize_factor
               + duration × asr_factor
```

- `diarize_factor` és `asr_factor` = az utolsó 15 futás **mediánja** (outlier-robusztus)
- Legalább **2 érvényes futás** szükséges a becslés megjelenítéséhez
- Az időtartam mérése **pydub**-bal történik

### `logs` DB tábla (statisztika)

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

## Modellválasztás

Az aktív ASR modell a `.env` `WHISPER_MODEL_DIR` változójával állítható:

| Állapot | `WHISPER_MODEL_DIR` értéke |
|---|---|
| Eredeti (alap) modell | `""` (üres) |
| Finomhangolt modell | `/srv/transcriber_app/finetune_output` |

A váltás és az **alkalmazás újraindítása** a `/finetune` aloldalon keresztül végezhető – terminálhoz való hozzáférés nélkül. Az újraindítás `os.execv()` alapú: a Python folyamat saját magát cseréli le, a `.env` újratöltődik, a systemd service folyamatosan fut.

---

## Modell finomhangolása

A `/finetune` aloldal kezeli a tanítóadat-gyűjtést, a betanítás indítását és a modellváltást.

**Jelenlegi állapot panel – 3 doboz:**

| Doboz | Tartalom |
|---|---|
| feltöltött tanítóminta | Darabszám |
| hanganyag összesen | Összes feltöltött hanganyag hossza (ó / p / mp formátumban) |
| finomhangolt modell | Kész ✓ ha `finetune_output/config.json` létezik |

> **Megjegyzés:** A betanított modell (`finetune_output/`) és a tanítóadatok (`training_data/`) **nem részei a publikus repónak** – ezek csak a szerveren léteznek lokálisan.

### Tanítóadat gyűjtése – egyedi feltöltés

A `/finetune` aloldalon egyenként tölthető fel hanganyag (max. 30 mp) és a hozzá tartozó javított leirat. Az oldal visszajátszót biztosít meghallgatáshoz mielőtt a szöveget begépelnéd.

### Tanítóadat gyűjtése – tömeges feltöltés

A `/finetune` oldal **Tömeges feltöltés** szekciójában egyszerre választható ki több hangfájl és a hozzájuk tartozó `.txt` leiratok. A párosítás fájlnév alapján automatikus (`hirek_001.wav` ↔ `hirek_001.txt`). A feltöltés sorban, egyenként fut a `/finetune/upload` endpointon.

**Hosszú fájlok automatikus darabolása feltöltéskor:**

Ha a feltöltött hanganyag hosszabb 35 másodpercnél (pl. 2–3 perces hírszegmens), a szerver automatikusan feldarabolja és N darab tanítómintaként menti el. A darabolás módja prioritás szerint:

| Prioritás | Módszer | Feltétel |
|---|---|---|
| 1. | **Adobe Audition XMP marker** | A fájlban `xmpDM:startTime` értékek találhatók |
| 2. | Leghosszabb csendpontok | Nincs marker, de van több bekezdés a txt-ben |
| 3. | Arányos időosztás | Nincs elég csendpont |

**Adobe Audition marker workflow:**

1. Auditionban helyezz el point markereket a hírsegmensek **közötti** elválasztó pontokra (N hírhez N−1 marker)
2. Exportálj MP3-t – az export dialógban az **„Include Markers and Other Metadata"** legyen bepipálva
3. Töltsd fel a `/finetune` → Tömeges feltöltés szekciójában a markerezett MP3-t és a bekezdéses txt-t
4. A szerver kiolvas minden `xmpDM:startTime` értéket (`f48000`-es időalapból ms-re konvertálva), és pontosan ott vágja a hangot

A visszajelzés megmutatja melyik módszerrel darabolódott: **„N mintára darabolva marker alapján"** vagy **„csend alapján"**.

**A szövegfájl elvárt formátuma (bekezdéses):**

```
Első hír szövege egy bekezdésben. Lehet több mondat is, de
csak ez a hírelem szerepeljen ebben a bekezdésben.

Második hír szövege. Üres sorral elválasztva az előzőtől.

Harmadik hír szövege.
```

Minden üres sorral elválasztott bekezdés = egy hírsegmens. N marker → N+1 szegmens → N+1 bekezdéssel párosítva.

### Tanítóadat lista

A feltöltött minták görgethet listában jelennek meg (fix magasság, ragasztott fejléc). Minden mintánál látható a hossz, a leirat előnézete, hogy melyik futásban lett felhasználva, és törölhető.

### Tanítóadat előkészítése offline (`split_news.py`)

Ha nem szeretnéd a nyers fájlt a szerverre feltölteni, a szerver-oldali `split_news.py` scripttel előre feldarabolhatod a hanganyagot. A script ugyanazt a logikát alkalmazza (marker > csend > arányos), és a kimenetét a Tömeges feltöltéssel viheted fel.

```bash
cd /srv/transcriber_app && source venv/bin/activate

# MP3 + bekezdéses txt → darabolás split_output/ mappába
python split_news.py hirek_0800.mp3 hirszoveg.txt

# Több fájl, egyedi kimeneti mappa
python split_news.py news/*.mp3 --out tanito_adatok/
```

### A betanítás menete

- Betanítás **kézzel indítható** – automatikus ütemezés alapból ki van kapcsolva
- Minden futás **az összes feltöltött mintán** végigmegy (nem csak az újakon) → a modell nem felejti el a korábbi anyagokat
- LoRA módszerrel tanít (`rank=8`, `alpha=32`, `target_modules: q_proj + v_proj`), majd `merge_and_unload()` után teljes modellként menti (~2.9 GB)

### Loss értelmezése

| Loss | Jelentés |
|---|---|
| 2.0+ | Alig tanult |
| 0.8–1.5 | Tanul, de kevés adat |
| 0.3–0.6 | Egészséges tartomány |
| 0.05 alatti | Túltanulás veszélye |

A loss futások között **nem hasonlítható össze közvetlenül** – több vagy változatosabb minta magasabb loss-t adhat. A tényleges minőség csak valódi hanganyagon mérhető.

### Tanítóadatok tárolási helye

| Adat | Elérési út |
|---|---|
| Hanganyagok | `/srv/transcriber_app/training_data/` (WAV, 16kHz mono) |
| Leiratok | `/srv/transcriber_app/logs/transcriber.db` → `training_data` tábla |
| Tanított modell | `/srv/transcriber_app/finetune_output/` (~2.9 GB) |

### `training_data` DB tábla

| Oszlop | Tartalom |
|---|---|
| `id` | Automatikus azonosító |
| `audio_fn` | Hangfájl neve a `training_data/` mappában |
| `transcript` | Javított leirat szövege |
| `uploaded_at` | Feltöltés időpontja |
| `duration_s` | Hanganyag hossza másodpercben |
| `used_in_run` | Melyik futásban lett először felhasználva (0 = még nem) |

### `finetune_runs` DB tábla

| Oszlop | Tartalom |
|---|---|
| `id` | Futás azonosítója |
| `started_at` / `finished_at` | Időbélyegek |
| `status` | `running` / `done` / `failed` |
| `samples_used` | Összes minta a futásban |
| `last_loss` | Utolsó tanítási lépés loss értéke |
| `output_dir` | Kimeneti könyvtár |
| `error_msg` | Hibaüzenet (ha `failed`) |

### Automatikus ütemezés (opcionális)

Alapból **ki van kapcsolva**. Bekapcsoláshoz `.env`-be:
```
FINETUNE_AUTO=1
FINETUNE_START_HOUR=22
FINETUNE_END_HOUR=6
```

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
| `DIARIZATION_DEVICE` | Diarizáció eszköze (`cuda`/`cpu`) | `cpu` |
| `DIARIZATION_BATCH_SIZE` | Pyannote batch méret | `8` |
| `ASR_BATCH_SIZE` | Hány chunk kerül egy GPU batch-be | `8` |
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
| 8 | 5 | ~11–12 GB | **alapértelmezett** |
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

### Leiratozás

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

### Finomhangolás (lokális)

| Végpont | Metódus | Leírás |
|---|---|---|
| `/finetune` | GET | Finomhangolás aloldal |
| `/finetune/upload` | POST | Egyedi tanítóadat feltöltése |
| `/finetune/data` | GET | Feltöltött minták listája (JSON) |
| `/finetune/data/<id>/delete` | POST | Minta törlése |
| `/finetune/audio/<filename>` | GET | Hanganyag letöltése |
| `/finetune/status` | GET | Státusz, log, futási előzmények |
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
  "total_samples": 43,
  "total_duration_s": 892,
  "model_ready": true,
  "model_dir": "/srv/transcriber_app/finetune_output",
  "active_model": "finetune",
  "active_label": "/srv/transcriber_app/finetune_output",
  "log_tail": ["STEP: 45/60 loss=0.3241", "EPOCH: 3/3 avg_loss=0.3189", "=== Kész ==="],
  "runs": [{
    "id": 8, "started_at": "...", "finished_at": "...",
    "status": "done", "samples_used": 43, "last_loss": 0.4280, "error_msg": null
  }]
}
```

`total_duration_s` – az összes feltöltött tanítóhanganyag hossza másodpercben (a UI óra/perc/mp formátumban jeleníti meg).

---

## Könyvtárstruktúra

```
leiratozo/
├── app.py                           # Flask szerver, queue, worker, becslés, modellváltás, finomhangolás API
├── convert.py                       # ffmpeg-alapú audio konvertálás
├── diarization.py                   # pyannote diarizáció
├── transcript_after_diarization.py  # Diarizáció utáni batch ASR pipeline
│                                    #   normalize → merge → split → ASR → merge_back
├── fast_whisper_transcribe.py       # Standalone: faster-whisper referencia
├── llm_refine.py                    # Standalone: Ollama szövegfinomítás
├── step1_merge_diar.py              # Standalone: diarizációs szegmensek egyesítése
├── step2_split_audio.py             # Standalone: audio felszeletelés
├── step3_transcribe.py              # Standalone: ASR pipeline
├── check_gpu.py                     # GPU ellenőrzés
├── transcriber.service              # systemd service unit
├── requirements.txt                 # Python függőségek
├── static/                          # Ikonok, statikus fájlok
└── templates/
    └── index.html                   # Web UI (queue, progress, statisztikák)
```

> A betanított modell (`finetune_output/`) és a tanítóadatok (`training_data/`) **nem részei a publikus repónak** – ezek csak a szerveren léteznek lokálisan.

### Futás közben létrehozott mappák (gitignore-ban)

```
queue/                    # Feltöltött, feldolgozásra váró fájlok
logs/                     # SQLite (transcriber.db)
cache/                    # HuggingFace model cache (~3 GB / modell)
templates/transcripts/    # Kész leiratok (30 perc után törlődnek)
training_data/            # Finomhangoláshoz feltöltött hanganyagok
finetune_output/          # Tanított modell (~2.9 GB)
split_output/             # split_news.py kimenete (feltöltés előtt)
```

### SQLite adatbázis (`logs/transcriber.db`)

| Tábla | Tartalom |
|---|---|
| `logs` | Leiratozási futások statisztikája |
| `queue` | Feldolgozási sor |
| `training_data` | Finomhangoláshoz feltöltött minták |
| `finetune_runs` | Finomhangolási futások előzményei |

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
- `max_new_tokens` maximum `444` lehet (`max_target_positions=448`, 4 decoder prompt token foglalt)

**Diarizáció hibával áll le**
- Ellenőrizd a `HUGGINGFACE_TOKEN` értékét
- HuggingFace oldalon el kell fogadni a `pyannote/speaker-diarization-3.1` feltételeit

**Hiányzó szöveg hosszú turn-öknél**
- Whisper maximális ablaka 30 mp. A `split_long_turns()` (25 mp-es határ) ezt automatikusan kezeli – ha régebbi futásnál merült fel, az újrafuttatás javítja.

**Finomhangolás: `No module named 'peft'`**
```bash
/srv/transcriber_app/venv/bin/pip install "peft>=0.11.0" "datasets>=2.20.0"
```

**Finomhangolás: `input_ids` hiba Whisper-rel**
- A `LoraConfig`-ban **nem szabad** `task_type=TaskType.SEQ_2_SEQ_LM` paramétert megadni – Whisper `input_features`-t vár, nem `input_ids`-t.

**`split_news.py`: nem talál elég csendpontot**
- A script figyelmeztet és arányos időosztásra vált – ellenőrizd a kimenet minőségét
- `SILENCE_DB` értékét emeld (-35, -30) ha a felvétel hangos környezetben készült
