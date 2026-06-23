from flask import Flask, request, render_template, send_from_directory, jsonify
import os, sys, subprocess, time, sqlite3, threading, re
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv, set_key
from pydub import AudioSegment

app = Flask(__name__)
BASE_DIR        = Path(__file__).resolve().parent
ENV_FILE        = BASE_DIR / ".env"
DEFAULT_CACHE   = BASE_DIR / "cache"
QUEUE_DIR       = BASE_DIR / "queue"

_active           = {"diar": None, "transcript": None}
_stop_requested   = False
_worker_lock      = threading.Lock()
_worker_running   = False

_progress = {
    "step": 0, "stepName": "", "percent": 0,
    "startTime": None, "estimatedTotal": 0,
    "estimatedDiarize": 0, "estimatedASR": 0, "detail": "",
}

status_messages = []


def _prompt_env_values():
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
    if not sys.stdin.isatty():
        return
    prompts = [
        ("HUGGINGFACE_TOKEN", "Add meg a HUGGINGFACE_TOKEN értékét:", None),
        ("PORT", "Add meg a Flask PORT értéket [58515]:", "58515"),
        ("HF_CACHE_DIR", f"Add meg a HF_CACHE_DIR értékét [{DEFAULT_CACHE}]:", str(DEFAULT_CACHE)),
        ("WHISPER_MODEL_ID", "WHISPER_MODEL_ID [Trendency/whisper-large-v3-hu]:", "Trendency/whisper-large-v3-hu"),
        ("WHISPER_MODEL_DIR", "Lokális WHISPER_MODEL_DIR (hagyd üresen):", ""),
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
                print("Kötelező."); continue
            if key == "PORT":
                try:
                    int(answer)
                except ValueError:
                    print("Számnak kell lennie."); continue
            break
        set_key(str(ENV_FILE), key, answer)
        os.environ[key] = answer
        updated = True
    if updated or ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)


_prompt_env_values()

UPLOAD_FOLDER      = BASE_DIR / "uploads"
LOG_FOLDER         = BASE_DIR / "logs"
TRANSCRIPTS_FOLDER = BASE_DIR / "templates" / "transcripts"
LOCK_FILE          = BASE_DIR / "transcriber.lock"
DB_FILE            = LOG_FOLDER / "transcriber.db"
PORT               = int(os.environ.get("PORT", "58515"))


def logprint(msg):
    print(msg, flush=True)
    if not status_messages:
        status_messages.append(msg)
    else:
        status_messages[-1] = msg


# ── DB ────────────────────────────────────────────────────────────────────────

def init_db():
    os.makedirs(LOG_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Statisztikai napló
    c.execute("""CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ip TEXT, filename TEXT, filetype TEXT, duration REAL,
        start_time TEXT, end_time TEXT, runtime REAL,
        diarize_time REAL DEFAULT 0, asr_time REAL DEFAULT 0
    )""")
    for col in ("diarize_time REAL DEFAULT 0", "asr_time REAL DEFAULT 0",
                "result_fn TEXT DEFAULT ''"):
        try:
            c.execute(f"ALTER TABLE logs ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    # Feldolgozási sor
    c.execute("""CREATE TABLE IF NOT EXISTS queue (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ip          TEXT,
        original_fn TEXT,
        stored_fn   TEXT,
        queued_at   TEXT,
        status      TEXT DEFAULT 'pending',
        started_at  TEXT,
        finished_at TEXT,
        result_fn   TEXT,
        error_msg   TEXT
    )""")
    # Induláskor a félbehagyott 'running' jobokat visszaállítjuk 'pending'-be
    c.execute("UPDATE queue SET status='pending', started_at=NULL WHERE status='running'")
    conn.commit()
    conn.close()


init_db()


# ── Queue DB helpers ──────────────────────────────────────────────────────────

def _set_job_status(job_id, status, **kw):
    cols = ["status=?"]
    vals = [status]
    for k in ("started_at", "finished_at", "result_fn", "error_msg"):
        if k in kw:
            cols.append(f"{k}=?")
            vals.append(kw[k])
    vals.append(job_id)
    conn = sqlite3.connect(DB_FILE)
    conn.execute(f"UPDATE queue SET {', '.join(cols)} WHERE id=?", vals)
    conn.commit()
    conn.close()


def _dequeue_next():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, ip, original_fn, stored_fn FROM queue WHERE status='pending' ORDER BY id LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "ip": row[1], "original_fn": row[2], "stored_fn": row[3]}
    return None


# ── Subprocess streaming ──────────────────────────────────────────────────────

def _stream_proc(proc, on_line=None):
    """Soronként olvassa stdout-ot; stderr-t thread-ben üríti."""
    stderr_buf = []

    def _drain():
        stderr_buf.extend(proc.stderr.read().splitlines())

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    lines = []
    for raw in proc.stdout:
        line = raw.rstrip().replace("\r", "")
        if line:
            lines.append(line)
            logprint(line)
            if on_line:
                on_line(line)
    proc.stdout.close()
    rc = proc.wait()
    t.join(timeout=10)
    return "\n".join(lines), "\n".join(stderr_buf), rc


def _on_asr_progress(line):
    m = re.match(r"PROGRESS:\s*(\d+)/(\d+)", line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        _progress["percent"] = int(22 + (done / total) * 78) if total else 22
        _progress["detail"]  = f"{done}/{total} turn"


# ── Statisztika és becslés ────────────────────────────────────────────────────

def get_step_factors():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""SELECT diarize_time/duration, asr_time/duration FROM logs
                 WHERE duration > 5 AND diarize_time > 0 AND asr_time > 0
                 ORDER BY id DESC LIMIT 15""")
    rows = c.fetchall()
    conn.close()
    n = len(rows)
    if n < 2:
        return None, None, n
    df = sorted(r[0] for r in rows)[n // 2]
    af = sorted(r[1] for r in rows)[n // 2]
    return df, af, n


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(job):
    """Feldolgoz egy queue-ban lévő job-ot. A worker thread hívja."""
    global _stop_requested
    job_id   = job["id"]
    ip       = job["ip"]
    filepath = QUEUE_DIR / job["stored_fn"]
    filename = job["original_fn"]
    file_ext = filename.rsplit(".", 1)[-1].lower()

    _set_job_status(job_id, "running", started_at=datetime.now().isoformat())
    status_messages.clear()
    _progress.update({"step": 0, "stepName": "", "percent": 0,
                       "startTime": None, "estimatedTotal": 0,
                       "estimatedDiarize": 0, "estimatedASR": 0, "detail": ""})

    diarize_time = 0
    asr_time     = 0

    try:
        os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
        start_time_sec     = time.time()
        _progress["startTime"] = start_time_sec

        # ── 1. Konvertálás ────────────────────────────────────────────────
        _progress.update({"step": 1, "stepName": "convert", "percent": 1, "detail": ""})
        logprint(f"[{filename}] Konvertálás kezdődik...")
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "convert.py"), str(filepath)],
            capture_output=True, text=True
        )
        if result.returncode != 0 or "Sikeres konvertálás" not in result.stdout:
            raise RuntimeError(f"Konvertálási hiba: {result.stderr[:300]}")
        _progress["percent"] = 5

        audio_wav = BASE_DIR / "audio.wav"
        if not audio_wav.exists():
            raise RuntimeError("audio.wav nem jött létre")

        duration     = len(AudioSegment.from_wav(str(audio_wav))) / 1000.0
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

        df, af, n_samples = get_step_factors()
        if df and af:
            est_d = duration * df
            est_a = duration * af
            est_t = 5 + est_d + est_a
            _progress.update({"estimatedTotal": int(est_t),
                               "estimatedDiarize": int(est_d),
                               "estimatedASR": int(est_a)})
            m, s = int(est_t // 60), int(est_t % 60)
            logprint(f"Becsült idő: kb. {m}:{s:02d} ({n_samples} futás alapján)")
        else:
            logprint(f"Becsléshez nincs elég adat ({n_samples}/2)")

        # ── 2. Diarizáció ─────────────────────────────────────────────────
        _progress.update({"step": 2, "stepName": "diarize", "percent": 5, "detail": ""})
        logprint("Diarizáció...")
        diar_start = time.time()
        diar_proc  = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "diarization.py")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        _active["diar"] = diar_proc
        diar_out, diar_err, diar_rc = _stream_proc(diar_proc)
        _active["diar"] = None
        diarize_time = time.time() - diar_start

        if _stop_requested:
            raise RuntimeError("Leállítva")
        if diar_rc != 0 or "Diarizáció kész" not in diar_out:
            raise RuntimeError(f"Diarizációs hiba: {diar_err[:300]}")
        _progress["percent"] = 22

        if _stop_requested:
            raise RuntimeError("Leállítva")

        # ── 3. ASR ────────────────────────────────────────────────────────
        _progress.update({"step": 3, "stepName": "transcribe", "percent": 22, "detail": ""})
        logprint("ASR leiratozás...")
        asr_start      = time.time()
        asr_proc       = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "transcript_after_diarization.py")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        _active["transcript"] = asr_proc
        asr_out, asr_err, asr_rc = _stream_proc(asr_proc, on_line=_on_asr_progress)
        _active["transcript"] = None
        asr_time = time.time() - asr_start

        if _stop_requested:
            raise RuntimeError("Leállítva")
        if asr_rc != 0 or "A leiratozott beszélgetés mentve" not in asr_out:
            raise RuntimeError(f"ASR hiba: {asr_err[:300]}")

        # ── Eredmény keresése ─────────────────────────────────────────────
        result_fn = None
        for line in asr_out.split("\n"):
            if "A leiratozott beszélgetés mentve" in line:
                result_fn = os.path.basename(line.split(": ")[-1].strip())
                break
        if not result_fn or not (TRANSCRIPTS_FOLDER / result_fn).exists():
            raise RuntimeError("Kimeneti fájl nem található")

        # ── Statisztika mentés ────────────────────────────────────────────
        end_time_sec = time.time()
        runtime      = end_time_sec - start_time_sec
        conn = sqlite3.connect(DB_FILE)
        conn.execute("""INSERT INTO logs
            (ip, filename, filetype, duration, start_time, end_time,
             runtime, diarize_time, asr_time, result_fn)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (ip, filename, file_ext, duration,
             datetime.fromtimestamp(start_time_sec),
             datetime.fromtimestamp(end_time_sec),
             runtime, diarize_time, asr_time, result_fn))
        conn.commit()
        conn.close()

        _progress["percent"] = 100
        m_rt, s_rt = int(runtime // 60), int(runtime % 60)
        logprint(f"Kész! ({m_rt}:{s_rt:02d})")

        _set_job_status(job_id, "done",
                        finished_at=datetime.now().isoformat(),
                        result_fn=result_fn)
        _cleanup_old_files()

    except Exception as e:
        for key in ("diar", "transcript"):
            p = _active.get(key)
            if p and p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
            _active[key] = None
        _set_job_status(job_id, "failed",
                        finished_at=datetime.now().isoformat(),
                        error_msg=str(e))
        logprint(f"Hiba: {e}")
    finally:
        # Queue fájl törlése feldolgozás után
        try:
            filepath.unlink(missing_ok=True)
        except Exception:
            pass


# ── Worker ────────────────────────────────────────────────────────────────────

def _queue_worker():
    global _worker_running, _stop_requested
    while True:
        _stop_requested = False
        job = _dequeue_next()
        if not job:
            with _worker_lock:
                _worker_running = False
            break
        _run_pipeline(job)


def _ensure_worker():
    global _worker_running
    with _worker_lock:
        if not _worker_running:
            _worker_running = True
            threading.Thread(target=_queue_worker, daemon=True).start()


# ── Segédfüggvények ───────────────────────────────────────────────────────────

def _cleanup_old_files():
    now = datetime.now()
    for f in os.listdir(TRANSCRIPTS_FOLDER):
        if not f.startswith(("final_transcription_", "final_text_")):
            continue
        try:
            ts = f.replace("final_transcription_","").replace("final_text_","").split(".")[0]
            for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S"):
                try:
                    ft = datetime.strptime(ts, fmt); break
                except ValueError:
                    continue
            else:
                continue
            if now - ft > timedelta(minutes=30):
                os.remove(TRANSCRIPTS_FOLDER / f)
        except Exception as e:
            print(f"Törlési hiba: {e}")


def _queue_position(job_id):
    """Visszaadja a job pozícióját a pending sorban (1-től indexelve)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM queue WHERE status='pending' ORDER BY id")
    ids = [r[0] for r in c.fetchall()]
    conn.close()
    try:
        return ids.index(job_id) + 1
    except ValueError:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "Nincs fájl kiválasztva!"})

    os.makedirs(QUEUE_DIR, exist_ok=True)
    filename  = file.filename
    prefix    = datetime.now().strftime("%Y%m%d%H%M%S%f")[:18]
    stored_fn = f"{prefix}_{filename}"
    file.save(QUEUE_DIR / stored_fn)

    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""INSERT INTO queue (ip, original_fn, stored_fn, queued_at, status)
                 VALUES (?,?,?,?,'pending')""",
              (request.remote_addr, filename, stored_fn, datetime.now().isoformat()))
    job_id = c.lastrowid
    c.execute("SELECT COUNT(*) FROM queue WHERE status='pending' ORDER BY id")
    position = c.fetchone()[0]
    conn.commit()
    conn.close()

    _ensure_worker()
    return jsonify({"queue_id": job_id, "position": position})


@app.route("/queue_status")
def queue_status():
    job_id = request.args.get("id", type=int)

    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()

    # Futó job
    c.execute("SELECT id, original_fn FROM queue WHERE status='running' LIMIT 1")
    running_row = c.fetchone()

    # Várakozók (sorrendben)
    c.execute("SELECT id, original_fn, queued_at FROM queue WHERE status='pending' ORDER BY id")
    pending_rows = c.fetchall()

    # Saját job
    my_job = None
    if job_id:
        c.execute("""SELECT id, status, result_fn, error_msg, original_fn
                     FROM queue WHERE id=?""", (job_id,))
        row = c.fetchone()
        if row:
            status_val = row[1]
            pos = None
            if status_val == "pending":
                pos = next((i+1 for i, r in enumerate(pending_rows) if r[0] == job_id), None)
            elif status_val == "running":
                pos = 0

            result_text = ""
            if status_val == "done" and row[2]:
                text_fn   = row[2].replace("final_transcription_","final_text_").replace(".json",".txt")
                text_path = TRANSCRIPTS_FOLDER / text_fn
                if text_path.exists():
                    with open(text_path, "r", encoding="utf-8") as f:
                        result_text = f.read()

            my_job = {
                "id":          row[0],
                "status":      status_val,
                "result_fn":   row[2],
                "error_msg":   row[3],
                "filename":    row[4],
                "position":    pos,
                "result_text": result_text,
            }

    conn.close()

    elapsed   = int(time.time() - _progress["startTime"]) if _progress.get("startTime") else 0
    estimated = _progress.get("estimatedTotal", 0)
    remaining = max(0, int(estimated - elapsed)) if estimated > 0 else 0

    return jsonify({
        "running": {"id": running_row[0], "filename": running_row[1]} if running_row else None,
        "pending_count": len(pending_rows),
        "queue": [{"id": r[0], "filename": r[1], "queued_at": r[2]} for r in pending_rows],
        "my_job": my_job,
        "progress": {
            "step":             _progress.get("step", 0),
            "stepName":         _progress.get("stepName", ""),
            "percent":          _progress.get("percent", 0),
            "elapsed":          elapsed,
            "estimatedTotal":   estimated,
            "estimatedDiarize": _progress.get("estimatedDiarize", 0),
            "estimatedASR":     _progress.get("estimatedASR", 0),
            "remaining":        remaining,
            "detail":           _progress.get("detail", ""),
            "currentStep":      status_messages[-1] if status_messages else "",
        },
    })


@app.route("/status")
def status():
    elapsed   = int(time.time() - _progress["startTime"]) if _progress.get("startTime") else 0
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


@app.route("/stats")
def stats():
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""SELECT id, filename, duration, diarize_time, asr_time, runtime, start_time,
                        COALESCE(result_fn, '') as result_fn
                 FROM logs WHERE duration > 5 ORDER BY id DESC LIMIT 20""")
    rows = c.fetchall()
    conn.close()

    runs = []
    for r in rows:
        result_fn  = r[7] or ""
        text_fn    = result_fn.replace("final_transcription_", "final_text_").replace(".json", ".txt")
        json_avail = bool(result_fn and (TRANSCRIPTS_FOLDER / result_fn).exists())
        text_avail = bool(text_fn   and (TRANSCRIPTS_FOLDER / text_fn).exists())
        runs.append({
            "id":         r[0], "filename": r[1],
            "duration_s": round(r[2] or 0, 1),
            "diarize_s":  round(r[3] or 0, 1),
            "asr_s":      round(r[4] or 0, 1),
            "total_s":    round(r[5] or 0, 1),
            "when":       r[6],
            "result_fn":  result_fn  if json_avail else "",
            "text_fn":    text_fn    if text_avail else "",
        })
    return jsonify({"count": len(rows), "runs": runs})


@app.route("/stop", methods=["POST"])
def stop():
    global _stop_requested
    _stop_requested = True
    killed = []
    for key in ("diar", "transcript"):
        p = _active.get(key)
        if p and p.poll() is None:
            try:
                p.kill(); killed.append(key)
            except Exception:
                pass
        _active[key] = None
    msg = f"Leállítva. Folyamatok: {killed}" if killed else "Nincs futó folyamat."
    return jsonify({"status": msg})


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(str(TRANSCRIPTS_FOLDER), filename)


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug_mode, threaded=True)
