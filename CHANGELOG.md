# Változásnapló

A jelentősebb funkcióváltozások időrendben. Részletes leírás: `README.md`.

## 2026-06-29 – Hullámforma-szerkesztő + tanítóadat minőség-osztályozás

### Hozzáadva

**Hullámforma-szerkesztő a diarizáló darabolóban**
- Beépített hullámforma-megjelenítés és -szerkesztő minden chunknál, **külső könyvtár nélkül**
  (Web Audio API + `<canvas>`) – offline szerveren is önálló.
- **Vágás** két sárga csúszóval: az elejéből/végéből levágható az elvágott szó / felesleges rész.
- **Környezeti hang hozzáadása**: a csúszók kifelé húzva környezeti hangot adnak a darab elejéhez/végéhez,
  mindig az **érintetlen eredeti felvételből** (nem destruktív forrás – vágás után is újra kibővíthető).
- **Lejátszás csak a két csúszó között** (`AudioBufferSourceNode`, mozgó lejátszófejjel), **play/pause**
  (onnan folytatódik, ahol megállt), a végpontnál automatikus megállás.
- **Scrub**: a hullámformára kattintva/húzva a lejátszási pont mozgatható (görgetés a hangban).
- A hullámformák a lista megnyitásakor **automatikusan betöltődnek** (a régi sima audio lejátszó eltávolítva).
- **Igazodó szövegdoboz**: a leirat-mező automatikusan a tartalom hosszához nő.
- **Egyesített mentés**: a „Mentés (szöveg + vágás)" egyetlen gombbal menti a leiratot és a vágást is.

**Hullámforma-szerkesztő a feltöltött tanítóadatoknál**
- A „Feltöltött tanítóadatok" táblában egy sorra kattintva a leirat mellett **hullámforma-szerkesztő** is
  megjelenik a már feltöltött (akár hosszú) minták **rövidítéséhez**. Itt csak rövidíteni lehet
  (nincs külön eredeti felvétel, a levágott rész nem állítható vissza).

**Tanítóadat minőség-osztályozás**
- Új **„Tanítóadat minőség"** panel a `/finetune` oldalon: minden minta ✅ jó / ⚠ gyanús / ⛔ árthat
  besorolást kap, indoklással (minőség-oszlop a táblázatban is).
- **Heurisztikus szint** (`training_quality.py`, GPU nélkül): cps (audio↔szöveg illeszkedés), hossz
  (>29s csonkolás, <1s), ismétlés/hallucináció, üres leirat, sok szám/írásjel, duplikátum.
- **Modell-audit szint** (`training_audit.py`, GPU, gombbal): az aktív Whisper modellel per-minta
  **WER** (átirat-eltérés) + **loss** – ezzel a már betanításra használt minták közül is kiderül,
  melyek viszik félre a tanulást. Az eredmény beépül az osztályozásba.
- **Forrás-túlsúly**: forrásonkénti megoszlás-riport + figyelmeztetés (`QUALITY_SOURCE_WARN_PCT`),
  és **forrás-sapka a tanításnál** (`FINETUNE_MAX_MIN_PER_SOURCE`) a domináns források alulmintavételezésére.

### Módosítva

- **Import (diarizáló daraboló) szinkronizálás**: ha egy **már importált** chunkon utólag javítod a
  szöveget vagy a vágást, a változás a tanítóadatba másolt példányon is érvényesül – **nem duplikál**
  (az import idempotens, a már importált chunkokat kihagyja).
- **Elavult audit törlése**: ha egy minta tartalma megváltozik (szöveg/vágás, diar és tábla egyaránt),
  a korábbi audit (WER/loss) törlődik az adott mintán.
- **Görgethető listák**: a minőség-panel forrás- és megjelölt-minta listái, valamint a
  „Korábbi darabolások" lista görgethetők.
- **Korábbi darabolások**: a listából eltávolítva a „Törlés" – a darabolások tartósan megmaradnak
  (`diar_split_sessions/`), nem veszhetnek el.

### Javítva

- A „Feltöltött tanítóadatok" tábla „Betöltés…"-nél ragadt: a `loadData()` initkor egy később deklarált
  `const`-ot ért el (temporal dead zone) – a hullámforma-állapot deklarációja a script tetejére került.
- A feltöltött minták lejátszó gombja nem reagált: az `onclick` string kulcsra `JSON.stringify` dupla
  idézőjelet adott, ami eltörte a HTML-attribútumot – string kulcsnál egyszeres idézőjelre javítva.

### Technikai

- **Új fájlok**: `training_quality.py`, `training_audit.py`.
- **DB (`training_data`) új oszlopok**: `source`, `audit_wer`, `audit_loss`, `audit_at` (automatikus migráció).
- **Új .env változók**: `FINETUNE_MAX_MIN_PER_SOURCE` (perc/forrás, 0=ki), `QUALITY_SOURCE_WARN_PCT` (alap 30).
- **Új API-végpontok**:
  - `GET  /finetune/quality` – minőség-összegzés, forráseloszlás, megjelölt minták
  - `POST /finetune/audit/start`, `GET /finetune/audit/status` – GPU modell-audit
  - `GET  /finetune/data/<id>/audio`, `POST /finetune/data/<id>/trim` – feltöltött minta hang + rövidítés
  - `GET  /finetune/diar-split/segment/<sid>/<idx>`, `POST /finetune/diar-split/trim/<sid>/<idx>` – hullámforma-ablak + vágás
