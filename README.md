# Leiratozó (Flask + Whisper + diarizáció)

Ez a mappa a leiratozó webalkalmazás megosztható verziója. Minden útvonal a projekt gyökeréhez viszonyított, a szükséges érzékeny értékeket futáskor kéri be és `.env`-be menti.

## Követelmények
- Python 3.11 (ajánlott)
- ffmpeg elérhető a PATH-ban (konvertáláshoz)
- CUDA képes GPU ajánlott (Whisper, pyannote), CPU-n is fut, de lassabb
- Internet a modellletöltésekhez (ha nincs lokális cache)

## Telepítés
```bash
cd /home/kkatai/leiratprojekt
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Futtatás
```bash
cd /home/kkatai/leiratprojekt
source venv/bin/activate
python app.py
```
Az első futáskor a program bekéri és a `.env`-be menti a szükséges értékeket:
- `HUGGINGFACE_TOKEN` (kötelező a diarizációhoz)
- `PORT` (Flask port, alapértelmezés 5000)
- `HF_CACHE_DIR` (modellcache helye, alapértelmezés: `cache/`)
- `WHISPER_MODEL_ID` (alapértelmezés: `Trendency/whisper-large-v3-hu`)
- `WHISPER_MODEL_DIR` (opcionális lokális modellkönyvtár)

Indítás után a felület: http://localhost:PORT

## Könyvtárstruktúra (fontosabbak)
- `app.py`: Flask szerver, a teljes pipeline vezérlése
- `convert.py`: ffmpeg alapú konvertálás `audio.wav`-ra
- `fast_whisper_transcribe.py`: gyors referencia leirat
- `diarization.py`: pyannote diarizáció, kimenet `diarization_result.json`
- `transcript_after_diarization.py`: végső leirat és szöveg mentése `templates/transcripts/`
- `templates/`: `index.html` és leiratok mappája
- `static/`: ikonok, `static/downloads/convert.exe`
- Üresen tartott futási mappák: `uploads/`, `logs/`, `chunks/`, `templates/transcripts/`

## Verziókezelésre ajánlott `.gitignore`
```
.env
venv/
__pycache__/
*.pyc
*.log
audio.wav
uploads/
logs/
chunks/
templates/transcripts/
cache/
```

## Docker (opcionális)
Van egy alap Dockerfile, de GPU-s futtatáshoz módosítani kell (CUDA runtime, megfelelő PyTorch+CUDA csomag).
