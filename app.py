from flask import Flask, request, render_template, send_from_directory, jsonify
import os, subprocess, time, sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv, set_key

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
DEFAULT_CACHE = BASE_DIR / "cache"

def _prompt_env_values():
    # Betöltjük a meglévő .env-et, ha van
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)

    prompts = [
        ("HUGGINGFACE_TOKEN", "Add meg a HUGGINGFACE_TOKEN értékét (diarizációhoz szükséges):", None),
        ("PORT", "Add meg a Flask PORT értéket [5000]:", "5000"),
        ("HF_CACHE_DIR", f"Add meg a HF_CACHE_DIR értékét [{DEFAULT_CACHE}]:", str(DEFAULT_CACHE)),
        ("WHISPER_MODEL_ID", "Add meg a WHISPER_MODEL_ID értékét [Trendency/whisper-large-v3-hu]:", "Trendency/whisper-large-v3-hu"),
        ("WHISPER_MODEL_DIR", "Opcionális lokális WHISPER_MODEL_DIR (hagyd üresen, ha nem kell):", ""),
    ]

    updated = False
    for key, question, default in prompts:
        current = os.environ.get(key, "").strip()
        if current:
            continue
        while True:
            answer = input(f"{question} ").strip()
            if not answer and default is not None:
                answer = default
            if key == "HUGGINGFACE_TOKEN" and not answer:
                print("A HUGGINGFACE_TOKEN kötelező a diarizációhoz.")
                continue
            if key == "PORT":
                try:
                    int(answer)
                except ValueError:
                    print("A PORT értékének számnak kell lennie.")
                    continue
            break
        # tárolás .env-ben és a futó környezetben
        set_key(str(ENV_FILE), key, answer)
        os.environ[key] = answer
        updated = True

    # újra betöltjük, hogy a set_key által írt értékek biztosan látszódjanak
    if updated or ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)

# gondoskodunk róla, hogy a futtatáshoz szükséges értékek .env-ben legyenek
_prompt_env_values()

UPLOAD_FOLDER = BASE_DIR / "uploads"
LOG_FOLDER = BASE_DIR / "logs"
TRANSCRIPTS_FOLDER = BASE_DIR / "templates" / "transcripts"
LOCK_FILE = BASE_DIR / "transcriber.lock"
DB_FILE = LOG_FOLDER / "transcriber.db"
PORT = int(os.environ.get("PORT", "5000"))

status_messages = []

def logprint(msg):
    print(msg)
    if len(status_messages) == 0:
        status_messages.append(msg)
    else:
        status_messages[-1] = msg

@app.route("/status")
def status():
    return jsonify({"currentStep": status_messages[-1] if status_messages else ""})

def init_db():
    os.makedirs(LOG_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ip TEXT,
        filename TEXT,
        filetype TEXT,
        duration REAL,
        start_time TEXT,
        end_time TEXT,
        runtime REAL
    )""")
    conn.commit()
    conn.close()

init_db()

def create_lock(ip):
    with open(LOCK_FILE, "w") as f:
        f.write(f"{ip}\n{time.time()}")

def is_locked():
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            data = f.readlines()
        if len(data) == 2:
            return data[0].strip(), datetime.fromtimestamp(float(data[1].strip()))
    return None, None

def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def calculate_audio_length(file_path):
    # ~32 kB/s becslés WAV-ra: elég a becsült időzítéshez
    return os.path.getsize(file_path) / 32000.0

def cleanup_old_files():
    now = datetime.now()
    for file in os.listdir(TRANSCRIPTS_FOLDER):
        if file.startswith(("final_transcription_", "final_text_", "final_refined_")):
            try:
                ts = file.replace("final_transcription_", "").replace("final_text_", "").replace("final_refined_", "")
                ts = ts.split(".")[0]
                file_time = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                if now - file_time > timedelta(minutes=30):
                    os.remove(TRANSCRIPTS_FOLDER / file)
            except Exception as e:
                print(f"Hiba a fájl törlésekor: {e}")

def get_average_factor():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT AVG(runtime/duration) FROM logs WHERE duration > 0")
    avg_value = c.fetchone()[0]
    conn.close()
    return avg_value

@app.route("/")
def index():
    ip, start_time = is_locked()
    return render_template("index.html",
                           locked=(ip is not None),
                           ip=ip,
                           start_time=start_time)

@app.route("/upload", methods=["POST"])
def upload():
    ip = request.remote_addr
    if is_locked()[0]:
        return jsonify({"error": "Már fut egy másik leiratozás!"})

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Nincs kiválasztva fájl!"})

    status_messages.clear()

    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(UPLOAD_FOLDER / f)

        create_lock(ip)
        start_time_sec = time.time()

        filename = file.filename
        file_ext = filename.rsplit(".", 1)[-1].lower()
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)

        logprint("Hangfájl konvertálása kezdődik...")
        result = subprocess.run(
            ["python3", str(BASE_DIR / "convert.py")],
            input=str(filepath),
            text=True,
            capture_output=True
        )
        if "Sikeres konvertálás" not in result.stdout:
            remove_lock()
            return jsonify({"error": "Konvertálási hiba!", "details": result.stderr})

        audio_converted = BASE_DIR / "audio.wav"
        if not os.path.exists(audio_converted):
            remove_lock()
            return jsonify({"error": "A konvertált audio.wav nem található!"})

        duration = calculate_audio_length(audio_converted)
        file_size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)

        avg_factor = get_average_factor()
        if avg_factor and avg_factor > 0:
            estimated_total_sec = duration * avg_factor
            minutes = int(estimated_total_sec // 60)
            seconds = int(estimated_total_sec % 60)
            logprint(f"Becsült feldolgozási idő: kb. {minutes} perc {seconds} mp")
        else:
            logprint("Nincs elegendő adat a becsült feldolgozási időhöz, folytatjuk...")

        # 1) Fast Whisper teljes anyag leiratozás INDÍTÁSA PÁRHUZAMOSAN
        logprint("Párhuzamos Fast Whisper leiratozás indítása...")
        fast_proc = subprocess.Popen(
            ["python3", str(BASE_DIR / "fast_whisper_transcribe.py")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # 2) Diarizáció
        logprint("Diarizáció kezdődik...")
        result = subprocess.run(
            ["python3", str(BASE_DIR / "diarization.py")],
            capture_output=True,
            text=True
        )
        if "Diarizáció kész" not in result.stdout:
            try:
                fast_proc.kill()
            except Exception:
                pass
            remove_lock()
            return jsonify({"error": "Diarizációs hiba!", "details": result.stderr})

        # 3) Várakozás a Fast Whisper befejezésére, legfeljebb 30 perc
        logprint("Várakozás a Fast Whisper befejezésére...")
        fast_proc.wait(timeout=1800)
        logprint("Fast Whisper kész.")

        # 4) Diarizált leiratozás + finomítás a fast_transcript alapján
        logprint("Leiratozás kezdődik (diarizált turn-ök)...")
        result = subprocess.run(
            ["python3", str(BASE_DIR / "transcript_after_diarization.py")],
            capture_output=True,
            text=True
        )
        if "A leiratozott beszélgetés mentve" not in result.stdout:
            remove_lock()
            return jsonify({"error": "Leiratozási hiba!", "details": result.stderr})

        final_filename = None
        for line in result.stdout.split('\n'):
            if "A leiratozott beszélgetés mentve" in line:
                final_filename = os.path.basename(line.split(': ')[-1].strip())
                break

        if not final_filename:
            remove_lock()
            return jsonify({"error": "Nem található a leiratozási kimeneti fájl!"})

        target_file = TRANSCRIPTS_FOLDER / final_filename
        if not os.path.exists(target_file):
            remove_lock()
            return jsonify({"error": f"A leiratozás eredményfájl nem található: {target_file}"})

        end_time_sec = time.time()
        runtime = end_time_sec - start_time_sec
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            INSERT INTO logs
            (ip, filename, filetype, duration, start_time, end_time, runtime)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ip,
            filename,
            file_ext,
            duration,
            datetime.fromtimestamp(start_time_sec),
            datetime.fromtimestamp(end_time_sec),
            runtime
        ))
        conn.commit()
        conn.close()

        cleanup_old_files()

        text_filename = final_filename.replace("final_transcription_", "final_text_").replace(".json", ".txt")
        text_filepath = TRANSCRIPTS_FOLDER / text_filename
        transcript_text = ""
        if os.path.exists(text_filepath):
            with open(text_filepath, "r", encoding="utf-8") as f:
                transcript_text = f.read()

        remove_lock()
        return jsonify({
            "success": "Leiratozás és párhuzamos Fast Whisper kész!",
            "filename": final_filename,
            "text": transcript_text,
            "file_info": {
                "name": filename,
                "size_mb": round(file_size_mb, 2),
                "duration_s": duration
            }
        })

    except Exception as e:
        remove_lock()
        return jsonify({"error": f"Váratlan hiba történt: {str(e)}"})
    # fontos: a fenti ágakban normál esetben a finally törölné a lockot,
    # de itt a sikeres visszatérés előtt már töröltük a lockot.
    # finally blokkra nincs szükség.
    
@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(str(TRANSCRIPTS_FOLDER), filename)

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug_mode)
