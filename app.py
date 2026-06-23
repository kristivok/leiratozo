from flask import Flask, request, render_template, send_from_directory, jsonify
import os, sys, subprocess, time, sqlite3, threading, re
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv, set_key
from pydub import AudioSegment

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
DEFAULT_CACHE = BASE_DIR / "cache"

_active = {"fast": None, "diar": None, "transcript": None}
_stop_requested = False

_progress = {
    "step": 0,
    "stepName": "",
    "percent": 0,
    "startTime": None,
    "estimatedTotal": 0,
    "estimatedDiarize": 0,
    "estimatedASR": 0,
    "detail": "",
}


def _prompt_env_values():
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
    if not sys.stdin.isatty():
        return
    prompts = [
        ("HUGGINGFACE_TOKEN", "Add meg a HUGGINGFACE_TOKEN értékét:", None),
        ("PORT", "Add meg a Flask PORT értéket [58515]:", "58515"),
        ("HF_CACHE_DIR", f"Add meg a HF_CACHE_DIR értékét [{DEFAULT_CACHE}]:", str(DEFAULT_CACHE)),
        ("WHISPER_MODEL_ID", "Add meg a WHISPER_MODEL_ID értékét [Trendency/whisper-large-v3-hu]:", "Trendency/whisper-large-v3-hu"),
        ("WHISPER_MODEL_DIR", "Opcionális lokális WHISPER_MODEL_DIR (hagyd üresen):", ""),
    ]
    updated = False
    for key, question, default in prompts:
        if os.environ.get(key, "").strip():
            continue
        while True:
            answer = input(f"{question} ").strip()
            if not answer and default is not None:
                answer = default
            if key == "HUGGINGFACE_TOKEN" and not answer:
                print("Kötelező.")
                continue
            if key == "PORT":
                try:
                    int(answer)
                except ValueError:
                    print("Számnak kell lennie.")
                    continue
            break
        set_key(str(ENV_FILE), key, answer)
        os.environ[key] = answer
        updated = True
    if updated or ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)


_prompt_env_values()

UPLOAD_FOLDER     = BASE_DIR / "uploads"
LOG_FOLDER        = BASE_DIR / "logs"
TRANSCRIPTS_FOLDER = BASE_DIR / "templates" / "transcripts"
LOCK_FILE         = BASE_DIR / "transcriber.lock"
DB_FILE           = LOG_FOLDER / "transcriber.db"
PORT              = int(os.environ.get("PORT", "58515"))

status_messages = []


def logprint(msg):
    print(msg)
    if not status_messages:
        status_messages.append(msg)
    else:
        status_messages[-1] = msg


# ── /status ──────────────────────────────────────────────────────────────────

@app.route("/status")
def status():
    elapsed  = int(time.time() - _progress["startTime"]) if _progress.get("startTime") else 0
    estimated = _progress.get("estimatedTotal", 0)
    remaining = max(0, int(estimated - elapsed)) if estimated > 0 else 0
    return jsonify({
        "currentStep":      status_messages[-1] if status_messages else "",
        "step":             _progress.get("step", 0),
        "stepName":         _progress.get("stepName", ""),
        "percent":          _progress.get("percent", 0),
        "elapsed":          elapsed,
        "estimatedTotal":   estimated,
        "estimatedDiarize": _progress.get("estimatedDiarize", 0),
        "estimatedASR":     _progress.get("estimatedASR", 0),
        "remaining":        remaining,
        "detail":           _progress.get("detail", ""),
    })


# ── /stats ────────────────────────────────────────────────────────────────────

@app.route("/stats")
def stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT id, filename, duration, diarize_time, asr_time, runtime, start_time
        FROM logs
        WHERE duration > 5
        ORDER BY id DESC
        LIMIT 20
    """)
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id":           r[0],
            "filename":     r[1],
            "duration_s":   round(r[2] or 0, 1),
            "diarize_s":    round(r[3] or 0, 1),
            "asr_s":        round(r[4] or 0, 1),
            "total_s":      round(r[5] or 0, 1),
            "when":         r[6],
        })
    return jsonify({"runs": result, "count": len(result)})


# ── DB ────────────────────────────────────────────────────────────────────────

def init_db():
    os.makedirs(LOG_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS logs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ip           TEXT,
        filename     TEXT,
        filetype     TEXT,
        duration     REAL,
        start_time   TEXT,
        end_time     TEXT,
        runtime      REAL,
        diarize_time REAL DEFAULT 0,
        asr_time     REAL DEFAULT 0
    )""")
    # Migration: add columns if they don't exist yet
    for col_def in ("diarize_time REAL DEFAULT 0", "asr_time REAL DEFAULT 0"):
        try:
            c.execute(f"ALTER TABLE logs ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


init_db()


# ── Subprocess streaming ──────────────────────────────────────────────────────

def _stream_proc(proc, on_line=None):
    """Soronként olvassa stdout-ot; stderr-t thread-ben üríti (deadlock megelőzés)."""
    stderr_buf = []

    def _drain():
        stderr_buf.extend(proc.stderr.read().splitlines())

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    stdout_lines = []
    for raw in proc.stdout:
        line = raw.rstrip().replace("\r", "")
        if line:
            stdout_lines.append(line)
            logprint(line)
            if on_line:
                on_line(line)
    proc.stdout.close()
    rc = proc.wait()
    t.join(timeout=10)
    return "\n".join(stdout_lines), "\n".join(stderr_buf), rc


def _on_asr_progress(line):
    """PROGRESS: X/Y → pontos % ASR lépésen belül."""
    m = re.match(r"PROGRESS:\s*(\d+)/(\d+)", line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        pct = int(22 + (done / total) * 78) if total > 0 else 22
        _progress["percent"] = pct
        _progress["detail"]  = f"{done}/{total} turn"


# ── Statisztika és becslés ────────────────────────────────────────────────────

def get_step_factors():
    """
    Visszaadja (diarize_factor, asr_factor, n) tuple-t.
    MEDIÁN-t használ az outlier-robusztusság miatt.
    Csak diarize_time > 0 és asr_time > 0 sorokat vesz figyelembe (utolsó 15 futás).
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT diarize_time / duration, asr_time / duration
        FROM logs
        WHERE duration > 5 AND diarize_time > 0 AND asr_time > 0
        ORDER BY id DESC
        LIMIT 15
    """)
    rows = c.fetchall()
    conn.close()
    n = len(rows)
    if n < 2:
        return None, None, n
    df = sorted(r[0] for r in rows)[n // 2]
    af = sorted(r[1] for r in rows)[n // 2]
    return df, af, n


# ── Lock / helpers ────────────────────────────────────────────────────────────

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


def calculate_audio_duration(wav_path):
    """Pontos időtartam pydub-bal (nem fájlméret-becslés)."""
    audio = AudioSegment.from_wav(str(wav_path))
    return len(audio) / 1000.0


def cleanup_old_files():
    """30 percnél régebbi leiratfájlokat töröl."""
    now = datetime.now()
    for file in os.listdir(TRANSCRIPTS_FOLDER):
        if not file.startswith(("final_transcription_", "final_text_")):
            continue
        try:
            ts = (file.replace("final_transcription_", "")
                      .replace("final_text_", "")
                      .split(".")[0])
            for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S"):
                try:
                    file_time = datetime.strptime(ts, fmt)
                    break
                except ValueError:
                    continue
            else:
                continue
            if now - file_time > timedelta(minutes=30):
                os.remove(TRANSCRIPTS_FOLDER / file)
        except Exception as e:
            print(f"Hiba a fájl törlésekor: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    ip, start_time = is_locked()
    return render_template("index.html",
                           locked=(ip is not None), ip=ip, start_time=start_time)


@app.route("/stop", methods=["POST"])
def stop():
    global _stop_requested
    _stop_requested = True
    killed = []
    for key in ("fast", "diar", "transcript"):
        proc = _active.get(key)
        if proc and proc.poll() is None:
            try:
                proc.kill()
                killed.append(key)
            except Exception:
                pass
        _active[key] = None
    remove_lock()
    msg = f"Leállítva. Folyamatok: {killed}" if killed else "Nincs futó folyamat."
    return jsonify({"status": msg})


@app.route("/upload", methods=["POST"])
def upload():
    global _stop_requested
    ip = request.remote_addr
    if is_locked()[0]:
        return jsonify({"error": "Már fut egy másik leiratozás!"})
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Nincs kiválasztva fájl!"})

    status_messages.clear()
    _stop_requested = False
    _progress.update({"step": 0, "stepName": "", "percent": 0,
                       "startTime": None, "estimatedTotal": 0,
                       "estimatedDiarize": 0, "estimatedASR": 0, "detail": ""})

    diarize_time = 0
    asr_time     = 0

    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(UPLOAD_FOLDER / f)

        create_lock(ip)
        start_time_sec      = time.time()
        _progress["startTime"] = start_time_sec

        filename  = file.filename
        file_ext  = filename.rsplit(".", 1)[-1].lower()
        filepath  = UPLOAD_FOLDER / filename
        file.save(filepath)

        # ── 1. Konvertálás ────────────────────────────────────────────────────
        _progress.update({"step": 1, "stepName": "convert", "percent": 1, "detail": ""})
        logprint("Hangfájl konvertálása kezdődik...")
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "convert.py"), str(filepath)],
            capture_output=True, text=True
        )
        if result.returncode != 0 or "Sikeres konvertálás" not in result.stdout:
            remove_lock()
            return jsonify({"error": "Konvertálási hiba!", "details": result.stderr})
        _progress["percent"] = 5

        audio_converted = BASE_DIR / "audio.wav"
        if not os.path.exists(audio_converted):
            remove_lock()
            return jsonify({"error": "A konvertált audio.wav nem található!"})

        # Pontos időtartam pydub-bal
        duration     = calculate_audio_duration(audio_converted)
        file_size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)

        # Becslés historikus adatokból (medián, per-lépéses)
        diarize_factor, asr_factor, n_samples = get_step_factors()
        if diarize_factor and asr_factor:
            est_diarize = duration * diarize_factor
            est_asr     = duration * asr_factor
            est_total   = 5 + est_diarize + est_asr
            _progress.update({
                "estimatedTotal":   int(est_total),
                "estimatedDiarize": int(est_diarize),
                "estimatedASR":     int(est_asr),
            })
            m, s = int(est_total // 60), int(est_total % 60)
            logprint(f"Becsült feldolgozási idő: kb. {m}:{s:02d} ({n_samples} futás alapján)")
        else:
            logprint(f"Becsléshez nincs elég adat ({n_samples}/2 futás) – következő futásoknál elérhető")

        # ── 2. Diarizáció ─────────────────────────────────────────────────────
        _progress.update({"step": 2, "stepName": "diarize", "percent": 5, "detail": ""})
        logprint("Diarizáció kezdődik...")
        diar_start = time.time()
        diar_proc  = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "diarization.py")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        _active["diar"]  = diar_proc
        diar_stdout, diar_stderr, diar_rc = _stream_proc(diar_proc)
        _active["diar"]  = None
        diarize_time     = time.time() - diar_start

        if _stop_requested:
            remove_lock()
            return jsonify({"error": "Leállítva."})
        if diar_rc != 0 or "Diarizáció kész" not in diar_stdout:
            remove_lock()
            return jsonify({"error": "Diarizációs hiba!", "details": diar_stderr})

        _progress["percent"] = 22

        if _stop_requested:
            remove_lock()
            return jsonify({"error": "Leállítva."})

        # ── 3. ASR ────────────────────────────────────────────────────────────
        _progress.update({"step": 3, "stepName": "transcribe", "percent": 22, "detail": ""})
        logprint("Leiratozás kezdődik (diarizált turn-ök)...")
        asr_start       = time.time()
        transcript_proc = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "transcript_after_diarization.py")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        _active["transcript"]  = transcript_proc
        transcript_stdout, transcript_stderr, transcript_rc = _stream_proc(
            transcript_proc, on_line=_on_asr_progress
        )
        _active["transcript"]  = None
        asr_time               = time.time() - asr_start

        if _stop_requested:
            remove_lock()
            return jsonify({"error": "Leállítva."})
        if transcript_rc != 0 or "A leiratozott beszélgetés mentve" not in transcript_stdout:
            remove_lock()
            return jsonify({"error": "Leiratozási hiba!", "details": transcript_stderr})

        # ── Mentés ────────────────────────────────────────────────────────────
        final_filename = None
        for line in transcript_stdout.split("\n"):
            if "A leiratozott beszélgetés mentve" in line:
                final_filename = os.path.basename(line.split(": ")[-1].strip())
                break

        if not final_filename or not os.path.exists(TRANSCRIPTS_FOLDER / final_filename):
            remove_lock()
            return jsonify({"error": "Nem található a leiratozási kimeneti fájl!"})

        end_time_sec = time.time()
        runtime      = end_time_sec - start_time_sec

        conn = sqlite3.connect(DB_FILE)
        c    = conn.cursor()
        c.execute("""
            INSERT INTO logs
              (ip, filename, filetype, duration, start_time, end_time,
               runtime, diarize_time, asr_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ip, filename, file_ext, duration,
              datetime.fromtimestamp(start_time_sec),
              datetime.fromtimestamp(end_time_sec),
              runtime, diarize_time, asr_time))
        conn.commit()
        conn.close()

        cleanup_old_files()

        text_filename = (final_filename
                         .replace("final_transcription_", "final_text_")
                         .replace(".json", ".txt"))
        text_filepath = TRANSCRIPTS_FOLDER / text_filename
        transcript_text = ""
        if os.path.exists(text_filepath):
            with open(text_filepath, "r", encoding="utf-8") as f:
                transcript_text = f.read()

        remove_lock()
        m_rt, s_rt = int(runtime // 60), int(runtime % 60)
        logprint(f"Kész! Teljes idő: {m_rt}:{s_rt:02d}")
        return jsonify({
            "success":   "Leiratozás kész!",
            "filename":  final_filename,
            "text":      transcript_text,
            "file_info": {
                "name":       filename,
                "size_mb":    round(file_size_mb, 2),
                "duration_s": round(duration, 1),
            },
        })

    except Exception as e:
        remove_lock()
        return jsonify({"error": f"Váratlan hiba: {str(e)}"})


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(str(TRANSCRIPTS_FOLDER), filename)


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug_mode, threaded=True)
