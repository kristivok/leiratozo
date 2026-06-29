# Leiratozó – Flask + Whisper + Diarizáció

Magyar nyelvű hangfájl-leiratozó webalkalmazás, amely **pyannote** beszélőazonosítást és **Whisper**-alapú átírást kombinál. Az eredmény beszélőnkénti, időbélyeges JSON és tiszta szöveges leirat.

Több felhasználó egyszerre is használhatja: a feldolgozás **sorban (queue)** zajlik, és bármely látogató valós időben láthatja az éppen zajló leiratozás előrehaladását.

> A jelentősebb funkcióváltozások időrendben: [`CHANGELOG.md`](CHANGELOG.md).

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
         < 2,5s turn-ök kihagyva (Whisper hallucinál rövid audión)
         Minden chunk → numpy array (16kHz float32)
         GPU batch: ASR_BATCH_SIZE chunk egyszerre
         GENERATE_KWARGS: language=hu, num_beams,
           no_repeat_ngram_size=5, repetition_penalty=1.3,
           max_new_tokens=444
         → PROGRESS: X/Y stdout-ra (batch-onként)
         clean_hallucination() – ismétlési hurkok levágása
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
- **Rövid szegmens szűrés** (`MIN_SEGMENT_S = 2.5`): ennél rövidebb turn-re Whisper nem fut – a modell rövid audión hallucinál (ismétlődő szótagok, értelmetlen karaktersorozatok).
- **Ismétlési hurok szűrés** (`clean_hallucination()`): a Whisper dekódoló néha hurokba kerül hosszú szegmensek végén. A függvény regex alapján keresi a 2-8 karakteres részletek 4×+ egymás utáni ismétlését, és ott csonkítja a szöveget. `no_repeat_ngram_size=5` és `repetition_penalty=1.3` csökkentik az előfordulást.

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

### Feltöltött tanítóadatok szerkesztése

A feltöltött tanítóadatok táblázatában minden sorra kattintva legördül egy szerkesztő panel, ahol a teljes leirat módosítható, **és egy hullámforma-szerkesztő is megjelenik a minta rövidítéséhez** (pl. a már feltöltött hosszú anyagok elejének/végének levágása). A szerkesztő ugyanaz, mint a diarizáló darabolónál (lejátszás a csúszók közt, scrub, igazodó szövegdoboz), de itt **csak rövidíteni** lehet – nincs külön eredeti felvétel, amiből környezeti hangot lehetne hozzáadni, így a levágott rész nem állítható vissza. A „Mentés (szöveg + vágás)" gomb a leiratot és a vágást is menti: a `POST /finetune/data/<id>/edit` a szöveget, a `POST /finetune/data/<id>/trim` pedig a WAV-ot vágja helyben és frissíti a hosszt.

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

### Tanítóadat gyűjtése – diarizáló daraboló

A `/finetune` oldal **Diarizáló daraboló** szekciója hosszú, többszereplős hanganyagokból (interjú, riport, kerekasztal) automatikusan állít elő tanítóadat-jelölteket.

**Workflow:**

1. Feltöltöd a hanganyagot (mp3, wav, m4a, …) a webes felületen
2. A rendszer háttérben:
   - Diarizációt futtat (**pyannote**, CPU-n, hogy ne ütközzön a fő pipeline GPU-használatával)
   - Whisper-rel leiratozza a teljes hanganyagot (szegmens-szintű timestampekkel)
   - Minden szegmenshez hozzárendeli a domináns bemondót (speaker purity alapján)
   - Mondathatárokon (`.!?`) és bemondóváltásoknál **12–25 mp-es darabokra** csomagolja az anyagot (max. 28,5 mp – Whisper 30 mp-es ablaka alatt marad)
3. Megjelennek a chunkok: minden darabhoz audio lejátszó, **hullámforma-szerkesztő** (vágás / környezeti hang) + szerkeszthető szöveg mező
4. Meghallgatod, hullámforma alapján pontosan vágod az elejét/végét, javítod a szöveget, majd importálod a tanítóadatba

**Átfedő (egymásra beszélős) szakaszok:**

A rendszer két rétegben szűri az átfedő szakaszokat:
- **pyannote overlap-zóna:** ha a szegmens bármely 0,3 mp-es részlete átfed egy detektált átfedő zónával → kihagyva
- **Speaker purity:** ha a szegmens kevesebb mint 80%-án egyetlen bemondó hallatszik → kihagyva

Az „Átfedő részek megtartása" jelölőnégyzettel mindkét szűrő kikapcsolható.

**Chunk törlése:**

Minden chunknál van egy piros „✕ Törlés" gomb – a törölt chunk elhalványodik és nem kerül importálásra. Visszaállítható az „↩ Visszaállítás" gombbal.

**Bemondónkénti tömeges törlés:**

A lista felett bemondónként megjelenik egy törlés gomb (pl. `✕ SPEAKER_01 törlése (8 db, 3.2 p)`). Hasznos telefonbeszélgetéseknél, ahol csak a stúdióban lévő bemondó hangja alkalmas betanításra.

**Hullámforma-szerkesztő – vágás és környezeti hang:**

Minden aktív chunknál a darab alatt **alapból megjelenik** egy beépített hullámforma-szerkesztő (nincs külön „sima" audio lejátszó, és nem kell külön gombra kattintani – a hullámformák a lista megnyitásakor automatikusan, egymás után betöltődnek). Ezzel pontosan állítható a darab eleje és vége: az elvágott szavakat és a felesleges részeket ki lehet vágni, illetve környezeti hangot lehet adni a darab elejéhez és végéhez, hogy a betanítandó minta tisztább legyen.

A szerkesztő **külső könyvtár nélkül** (Web Audio API + `<canvas>`) működik, így offline szerveren is teljesen önálló:

- **Hullámforma:** a darab köré a rendszer az **eredeti felvételből** kivág egy ±6 mp-es ablakot, és kirajzolja a hullámformát. A két szaggatott szürke vonal a darab eredeti határait jelzi.
- **Két sárga csúszó:** befelé húzva levágja a felesleges/elvágott részt, kifelé húzva környezeti hangot ad az elejéhez/végéhez (az eredeti felvételből). A csúszók alatti címkék élőben mutatják a változást: `eleje: +0.40s környezet` (zöld) vagy `vége: −0.25s vágva` (narancs), valamint az eredő hosszt.
- **▶ Lejátszás / ⏸ Szünet:** a lejátszás **csak a két csúszó között** szól (pontos `AudioBufferSourceNode`, mozgó lejátszófejjel) – a vágóponton kívüli részeket sosem játssza le. Szüneteltethető (onnan folytatódik, ahol megállt), és a végpontnál automatikusan megáll. A csúszók mozgatása leállítja a lejátszást.
- **Húzható lejátszófej (scrub):** a hullámformára kattintva / húzva a lejátszási pont mozgatható (görgetés a hangban). A pozíció a két csúszó közé van korlátozva; ha húzás közben épp ment a lejátszás, a húzás végén az új pontról folytatódik.
- **Igazodó szövegdoboz:** a leirat szerkesztőmezője automatikusan a szöveg hosszához nő (nem kell kézzel nagyítani az olvasáshoz).
- **↺ Eredeti:** visszaállítja a csúszókat a darab eredeti határaira.
- **Mentés:** nincs külön „vágás mentése" gomb – a darab alatti **„Mentés (szöveg + vágás)"** gomb egyszerre menti a leiratot és (ha a csúszók elmozdultak) a beállított vágást is. Vágáskor a chunk WAV újravágódik, és csak az adott darab hullámformája töltődik újra, a többi érintetlen marad.

**Fontos – nem destruktív forrás:** a vágás mindig az érintetlen **eredeti hangfájlból** készül (amit a session mappa megőriz), nem a már levágott chunkból. Így vágás után is bármikor újra ki lehet bővíteni a darabot a környezeti hangba. Mentéskor frissül a chunk `start`/`end`/`duration_s` értéke a `state.json`-ban. Ha a chunk **már importálva lett**, a mentés a `training_data/` mappába másolt példányt és a DB `duration_s` mezőjét is újravágja/frissíti, hogy a tanítóadat konzisztens maradjon.

**Importálás:**

Az „Összes importálása tanítóadatba" gomb az összes aktív (nem törölt, nem üres szövegű) chunkot beírja a `training_data` táblába és átmásolja a WAV fájlt a `training_data/` mappába.

**Nem duplikál:** az import chunkonként figyeli az `imported` flaget (a session `state.json`-jában), és a már importáltakat kihagyja – újra rányomva sem keletkezik kettőzött minta.

**Javítás szinkronizálása:** ha egy **már importált** chunkon utólag javítod a **szöveget** vagy a **vágást** (és mented), a változás a tanítóadatba másolt példányon is érvényesül (a `training_data` leiratát / WAV-ját és hosszát frissíti, nem hoz létre új mintát). Mivel ilyenkor a tartalom megváltozik, a korábbi **modell-audit eredménye (WER/loss) törlődik** az adott mintán (újra kell auditálni). Ugyanez igaz a „Feltöltött tanítóadatok" tábla szerkesztésére/vágására is.

**ZIP letöltés:**

A „↓ ZIP letöltése" gomb az összes chunk WAV + TXT párját ZIP archívumban tölti le – offline javításhoz vagy más eszközzel való feldolgozáshoz.

**Korábbi darabolások:**

Az oldal alján megjelenik a korábbi session-ök **görgethető** listája. A „Megnyitás" gombbal bármelyik visszatölthető. A darabolások **tartósan megmaradnak** (a `diar_split_sessions/<sid>/` mappában, újraindítás után is) – a listából **nincs törlés**, hogy egy korábbi darabolás se vesszen el.

**Parancssori használat (`diar_sentence_split.py`):**

```bash
source venv/bin/activate

# Alaphasználat
python diar_sentence_split.py --audio riport.mp3 --out tananyag/

# Átfedő részek megtartásával
python diar_sentence_split.py --audio vita.mp3 --out tananyag/ --keep-overlap

# Egyedi célhossz
python diar_sentence_split.py --audio hosszu.mp3 --out tananyag/ --target-min 20 --target-max 28
```

Kimenet: `tananyag/chunk_0001_SPEAKER_00.wav` + `tananyag/chunk_0001_SPEAKER_00.txt` párok, és `manifest.json`.

Javítás után batch import:

```bash
python diar_batch_import.py --dir tananyag/
python diar_batch_import.py --dir tananyag/ --dry-run    # csak előnézet
python diar_batch_import.py --dir tananyag/ --min-dur 3  # min 3 mp-es chunkok
```

**Session munkamappa felépítése:**

```
diar_split_sessions/<session_id>/
├── <eredeti_fajl>.mp3           # feltöltött hanganyag
├── diarization_result.json      # pyannote kimenet (cachelve)
├── manifest.json                # chunk metaadat lista
├── state.json                   # session állapot (chunks: start/end/duration_s, deleted, imported, imported_dest)
├── chunk_0000_SPEAKER_00.wav
├── chunk_0000_SPEAKER_00.txt
├── chunk_0001_SPEAKER_01.wav
└── ...
```

> A `diar_split_sessions/` mappa **nem része a publikus repónak**.

### Tanítóadat minőség-osztályozás

A `/finetune` oldal **Tanítóadat minőség** kártyája megjelöli, mely minták javíthatják a modellt
(✅ jó), és melyek vihetik félre a tanulást (⛔ árthat / ⚠ gyanús). A hangsúly a „biztosan árt"
eseteken van, hogy egy hibás vagy túlreprezentált anyag ne tanítsa félre a modellt.

**Két szint:**

1. **Heurisztikus (azonnali, GPU nélkül)** – `training_quality.py`. Szöveg + időtartam alapú jelek:
   - **cps (karakter/másodperc)**: extrém érték → audio↔szöveg eltérés (pl. `cps=73` = a szöveg sokszorosa annak, ami elhangozhat → ⛔).
   - **hossz**: `>29s` (Whisper 30s-os ablakán túl → csonkolt tanítás) vagy `<1s` → ⛔.
   - **ismétlési hurok / hallucináció** (a `clean_hallucination` logikájával), **üres leirat**, **sok szám/írásjel**.
   - **duplikátum**: ugyanaz a leiratszöveg többször → ⚠.
2. **Modell-audit (GPU, gombbal indítható)** – `training_audit.py`. Az **aktív** Whisper modellel végigmegy
   minden mintán, és per-mintára **WER**-t (a modell átirata vs. a tárolt referencia) és **loss**-t mér.
   Magas WER → valószínű hibás címke; magas loss → a modell a tanítás ellenére sem illeszti.
   Ezzel a **már betanításra használt** minták közül is kiderül, melyek vitték félre a tanulást.
   Az eredmény a `training_data.audit_wer / audit_loss / audit_at` oszlopokba kerül, és beépül az osztályozásba.
   A finomhangolás alatt nem indítható (GPU-ütközés). Az audit **háttérben fut**, és a terminál-logja
   (mint a finetune logja) **oldalfrissítés után is** mutatja a folyamatot – bármikor töltöd újra az oldalt,
   az éppen futó audit állapotát látod.

A jelölés **tájékoztató** – a tanítást nem módosítja, a gyenge mintákat a táblázatban kézzel törölheted.

**A jelölés indoka és javítása mintánként, a táblázatban:** a minőség-panel csak az összegzést, a
forráseloszlást és az auditot tartalmazza; a **megjelölés indoka („log") és a javítás közvetlenül az
egyes mintáknál**, a **„Feltöltött tanítóadatok"** táblában jelenik meg – a leirat alatt látszik az ok
(pl. *„túl hosszú (33.3s > 29s) · emelt WER (53%)"*), a sorban pedig egy **`✓`** gomb a kézi „jó"
jelöléshez. A táblázat fölötti **„Csak a megjelöltek (⛔/⚠)"** kapcsolóval csak a jelöltek listázhatók
(a panel „Megjelöltek mutatása a táblában" gombja is ezt kapcsolja be).

**Hibás jelölés kézi levétele:** ha egy minta tévesen kapott ⛔/⚠ címkét (pl. „emelt WER", de a leirat
valójában jó), a sor **`✓`** gombjával (vagy a szerkesztő-sorban „✓ Jónak jelölés") kézzel jónak
jelölheted. Ez **felülírja** az automatikus besorolást (heurisztika ÉS audit), és **a következő audit
után is megmarad** – a minta `✓ ellenőrizve` jelölést kap. A sor **`↩`** gombjával bármikor visszavonható.
(Tárolás: `training_data.quality_ok`; végpont: `POST /finetune/data/<id>/quality-ok`.)

**Forrás-túlsúly és sapka:**

A minták forrását (`source` oszlop) a feltöltéskor tároljuk (diar session / feltöltött fájlnév); a
régi mintákat egyszeri best-effort visszatöltés tölti ki (leiratszöveg-egyezés a diar session-ökkel,
különben fájlnévből). A panel forrásonkénti megoszlást mutat, és figyelmeztet, ha egy forrás a
hanganyag `QUALITY_SOURCE_WARN_PCT`%-át (alap: 30%) meghaladja. A `FINETUNE_MAX_MIN_PER_SOURCE`
(perc/forrás, `0`=ki) bekapcsolásával a `finetune_run.py` **alulmintavételezi** a domináns forrásokat,
hogy egyetlen anyag se tanítsa félre a modellt (ismeretlen forrásúak nincsenek korlátozva).

---

### Tanítóadat-gyűjtési útmutató (jó gyakorlatok)

**Fő elv:** a finomhangolás azt tanulja meg, amit adsz neki. Két cél: (1) a modell legyen jó azon, amit
élesben átírsz → az adat **tükrözze a valós használatot** (a ti hangjaitok, mikrofonjaitok, szókincsetek);
(2) ne torzuljon → **változatosság** kell. A legfontosabb messze a **címke pontossága** (a szöveg pontosan
illik a hanghoz) – egy hibás címke többet ront, mint amennyit sok jó minta javít.

**Milyen mintákat töltsünk fel – diverzitás.** Törekedj változatosságra:
- **beszélők**: több műsorvezető + sok különböző vendég/betelefonáló;
- **akusztikai körülmények**: stúdió, **telefon**, terepi/zajos – pont amilyenek élesben előfordulnak;
- **témák/szókincs**: minél többféle, hogy a nevek, szakszavak sokféle kontextusban szerepeljenek.
- Technikai: hossz **2–25 mp** (29 fölött csonkol), **cps 8–20**, nincs elvágott szó a széleken, tiszta,
  egy beszélős szakasz.

**Mi ISMÉTLŐDJÖN (hasznos ismétlés):** visszatérő tulajdonnevek/szakszavak **különböző mondatokban**
(így tanulja meg helyesen leírni őket); a valós akusztikai körülmények bő lefedése; **egységes** központozási
és számírási stílus mindenhol.

**Mi NE ismétlődjön (káros ismétlés):**
- **szó szerinti duplikátum** (ugyanaz a klip/mondat sokszor) – nincs tanulási értéke (a panel ⚠-vel jelzi);
- **boilerplate intro/outro/szignál** (pl. állandó köszöntés, szponzor-szöveg) – a modell „odahallucinálja",
  ahova nem való; ezekből keveset tarts, vagy vágd ki;
- **egyetlen forrás vagy hang túlsúlya** (lásd forrás-sapka).

**Fix műsorvezetők aránya.** Használd őket bőven (élesben is ők dominálnak; az ASR-finomhangolás nem
beszélő-felismerés, így a beszélő-túlsúly kevésbé veszélyes, mint egy hibás címke), de hagyj helyet a
vendégeknek/telefonnak, hogy a modell általánosítson:

| Tartomány | Ökölszabály |
|---|---|
| Műsorvezetők összesen | a korpusz nagyobb része lehet (~50–70%), ha élesben is ennyi |
| Egyetlen műsorvezető | ne menjen ~40% fölé egyedül |
| Vendég / telefon / terepi / egyéb | maradjon ~30–40% |
| Egyetlen felvétel (forrás) | sapkázd (pl. max 10–15 perc/forrás) |

**Rossz minőségű / telefonos hang = értékes.** A telefonos, zajos hang **nem csak megengedett, hanem
kifejezetten hasznos** – az alapmodell ezen a leggyengébb, így itt javít a legtöbbet a finomhangolás –,
**amíg a leirat pontos** és a beszéd érthető. A „rossz akusztika" (telefon/zaj) jó; a „rossz minta"
(bizonytalan/hibás címke, érthetetlen vagy átfedő beszéd) árt. Kivehetetlen szót **ne tippelj** – vágd ki
vagy hagyd ki. Ezek a minták tipikusan **magasabb WER/loss**-t kapnak: ez **normális, nem ok a kidobásra** –
hallgasd meg, és ha a leirat helyes, tartsd meg (akár „✓ jó"-ra jelölve).

**A loss/WER értelmezése a gyűjtésnél.**
- **loss < 0,1**: a szöveg pontosan illik a hanghoz → **tiszta, korrekt** minta. De a modell ezt **már tudja**,
  ezért a **tanulási értéke kicsi** (a gradiens ~0). **Nem haszontalan** – stabilizál, megakadályozza a
  felejtést a gyakori/könnyű eseteken –, csak nem innen jön a fejlődés. **Ne töröld** őket.
- A tényleges fejlődés a **nehéz, de helyesen címkézett** mintákból jön (ritka nevek, telefon, zaj, akcentus)
  – ezek közepes–magasabb lossúak. A javuláshoz: **javítsd/töröld a hibás címkéket** (magas WER) és **hozz be
  több nehéz, helyes mintát**.
- Ha sok a 0,1 alatti minta, az azt jelzi, hogy az alapmodell már jó a tipikus anyagotokon; érdemes több
  nehéz/változatos anyaggal bővíteni (nem a könnyűeket törölni). Ha a 0,1 alattiak közt sok a **rövid,
  triviális** klip („Köszönöm", „Jó napot"), az jelzés a korpusz kiegyensúlyozására.

---

### A betanítás menete

- Betanítás **kézzel indítható** – automatikus ütemezés alapból ki van kapcsolva
- Minden futás **az összes feltöltött mintán** végigmegy (nem csak az újakon) → a modell nem felejti el a korábbi anyagokat
- LoRA módszerrel tanít (`rank=32`, `alpha=64`, `target_modules: q_proj + k_proj + v_proj + out_proj`), majd `merge_and_unload()` után teljes modellként menti (~2.9 GB)
- Az encoder súlyai le vannak fagyasztva – csak a decoder LoRA rétegei tanulnak, ez ~50%-kal csökkenti a VRAM-igényt a backprop során
- Gradient accumulation támogatás: kis batch méret esetén több mini-batch gradiensét összegzi egy optimizer lépés előtt (`FINETUNE_GRAD_ACCUM` env var)
- Alapértelmezett konfiguráció: `batch=4`, `accum=1`, ~9–10 GB VRAM, ~210 optimizer lépés 3 epoch és 278 minta esetén

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
| `source` | Minta forrása túlsúly-méréshez (`diar:<fájl>` / `feltöltés:<név>`) |
| `audit_wer` | Modell-audit: WER a modell átirata vs. referencia (NULL = nem auditált) |
| `audit_loss` | Modell-audit: per-minta loss |
| `audit_at` | Audit időpontja |
| `quality_ok` | Kézzel „jónak" jelölve (1) – felülírja az auto-besorolást, túléli az auditot |

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
| `FINETUNE_GRAD_ACCUM` | Gradient accumulation lépések | `1` |
| `FINETUNE_LR` | Tanulási ráta | `1e-4` |
| `FINETUNE_LORA_RANK` | LoRA rang | `32` |
| `FINETUNE_LORA_ALPHA` | LoRA alpha | `64` |
| `FINETUNE_MAX_STEPS` | Max lépések (`0`=korlátlan) | `0` |
| `FINETUNE_MAX_MIN_PER_SOURCE` | Forrásonkénti felső sapka a tanításnál percben (`0`=ki) | `0` |
| `QUALITY_SOURCE_WARN_PCT` | Túlsúly-figyelmeztetés küszöbe (egy forrás a hanganyag ennyi %-a fölött) | `30` |

### VRAM és pontosság – ASR (átirat)

| ASR_BATCH_SIZE | ASR_NUM_BEAMS | Becsült VRAM | Jelleg |
|---|---|---|---|
| 8 | 5 | ~11–12 GB | **alapértelmezett** |
| 8 | 3 | ~10 GB | egyensúly |
| 16 | 1 | ~5–6 GB | greedy, gyors |

### VRAM és pontosság – finomhangolás

Az encoder le van fagyasztva, csak a decoder LoRA súlyai tanulnak.

| FINETUNE_BATCH_SIZE | FINETUNE_LORA_RANK | Becsült VRAM | Jelleg |
|---|---|---|---|
| 4 | 32 | ~9–10 GB | **alapértelmezett** – egyensúly |
| 1 | 32 | ~4–5 GB | kis VRAM, lassabb |
| 4 | 8 | ~6–7 GB | kevesebb tanítható param |
| 8 | 32 | ~14 GB | nagy VRAM, de kevesebb optimizer lépés |

> Ha VRAM hiba (OOM) jelentkezik, csökkentsd a `FINETUNE_BATCH_SIZE`-t vagy a `FINETUNE_LORA_RANK`-ot `.env`-ben.

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
| `/finetune/data/<id>/edit` | POST | Minta leiratának szerkesztése |
| `/finetune/data/<id>/delete` | POST | Minta törlése |
| `/finetune/data/<id>/audio` | GET | Minta hangja ID alapján (hullámforma-szerkesztőhöz) |
| `/finetune/data/<id>/trim` | POST | Feltöltött minta WAV-jának rövidítése helyben (`{start, end}`) |
| `/finetune/data/<id>/quality-ok` | POST | Minta kézi „jó" jelölése / visszavonása (`{ok}`) – felülírja az auto-besorolást |
| `/finetune/audio/<filename>` | GET | Hanganyag letöltése |
| `/finetune/quality` | GET | Minőség-osztályozás: összegzés, forráseloszlás, megjelölt minták |
| `/finetune/audit/start` | POST | GPU modell-audit indítása (per-minta WER + loss) |
| `/finetune/audit/status` | GET | Audit állapota, log, auditált/összes |
| `/finetune/status` | GET | Státusz, log, futási előzmények |
| `/finetune/trigger` | POST | Finomhangolás kézi indítása |
| `/finetune/stop` | POST | Futó finomhangolás leállítása |
| `/finetune/activate` | POST | Finomhangolt modell aktiválása (.env) |
| `/finetune/revert` | POST | Visszaállítás az eredeti modellre (.env) |

### Diarizáló daraboló

| Végpont | Metódus | Leírás |
|---|---|---|
| `/finetune/diar-split/start` | POST | Hangfájl feltöltése, darabolás indítása |
| `/finetune/diar-split/status/<sid>` | GET | Feldolgozás állapota, log, chunkok |
| `/finetune/diar-split/sessions` | GET | Korábbi session-ök listája |
| `/finetune/diar-split/audio/<sid>/<fn>` | GET | Chunk WAV fájl kiszolgálása |
| `/finetune/diar-split/segment/<sid>/<idx>` | GET | Hullámforma-ablak az eredeti hangból (±`pad` mp, WAV 16kHz mono). Az ablak/chunk határokat `X-Window-Start/End` és `X-Chunk-Start/End` fejlécekben adja vissza |
| `/finetune/diar-split/trim/<sid>/<idx>` | POST | Chunk újravágása az eredeti hangból (`{start, end}` abszolút mp). Importált chunknál a `training_data/` példányt és a DB hosszt is frissíti |
| `/finetune/diar-split/save/<sid>/<idx>` | POST | Egyedi chunk átirat mentése |
| `/finetune/diar-split/toggle-delete/<sid>` | POST | Chunk(ok) törlése / visszaállítása |
| `/finetune/diar-split/import/<sid>` | POST | Aktív chunkok importálása tanítóadatba |
| `/finetune/diar-split/download/<sid>` | GET | ZIP letöltés (WAV + TXT párok) |
| `/finetune/diar-split/delete/<sid>` | POST | Session törlése |

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
├── diar_sentence_split.py           # Diarizáló daraboló (parancssori)
├── diar_batch_import.py             # Batch import javított chunkokból
├── finetune_chunks.py               # Finomhangoláshoz: chunk előkészítő
├── finetune_prepare.py              # Finomhangoláshoz: mappa előkészítő
├── finetune_run.py                  # Finomhangolás futtatója (LoRA) – forrás-sapkával
├── finetune_transcript.py           # Finomhangoláshoz: leirat-előkészítő
├── training_quality.py              # Tanítóadat minőség-heurisztikák (cps, hossz, ismétlés, WER…)
├── training_audit.py                # GPU modell-audit: per-minta WER + loss a tanítóadaton
├── split_news.py                    # Hírszegmensek darabolása feltöltés előtt
├── check_gpu.py                     # GPU ellenőrzés
├── transcriber.service              # systemd service unit
├── requirements.txt                 # Python függőségek
└── templates/
    ├── index.html                   # Web UI (queue, progress, statisztikák)
    └── finetune.html                # Finomhangolás aloldal
```

> A betanított modell (`finetune_output/`) és a tanítóadatok (`training_data/`) **nem részei a publikus repónak** – ezek csak a szerveren léteznek lokálisan.

### Futás közben létrehozott mappák (gitignore-ban)

```
queue/                    # Feltöltött, feldolgozásra váró fájlok
logs/                     # SQLite (transcriber.db)
cache/                    # HuggingFace model cache (~3 GB / modell)
templates/transcripts/    # Kész leiratok (30 perc után törlődnek)
diar_split_sessions/      # Diarizáló daraboló ideiglenes session mappái
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

**Üres szegmens a kimenetben**
- A 2,5 mp-nél rövidebb turn-ök szándékosan üresek – Whisper ezeken hallucinál. Jellemzően rövid visszajelzések (`Igen`, `Értem`, `Jó`) esnek ki.

**Szöveg közepén csonkított mondat**
- A `clean_hallucination()` levágta az ismétlési hurkot. Az előtte lévő szövegrész helyes, a hiányzó rész az audión valószínűleg homályos/zajos volt.

**Finomhangolás: `No module named 'peft'`**
```bash
/srv/transcriber_app/venv/bin/pip install "peft>=0.11.0" "datasets>=2.20.0"
```

**Finomhangolás: `input_ids` hiba Whisper-rel**
- A `LoraConfig`-ban **nem szabad** `task_type=TaskType.SEQ_2_SEQ_LM` paramétert megadni – Whisper `input_features`-t vár, nem `input_ids`-t.

**`split_news.py`: nem talál elég csendpontot**
- A script figyelmeztet és arányos időosztásra vált – ellenőrizd a kimenet minőségét
- `SILENCE_DB` értékét emeld (-35, -30) ha a felvétel hangos környezetben készült
