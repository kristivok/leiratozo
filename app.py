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


# ═══════════════════════════════════════════════════════════════════════════════
# FINOMHANGOLÁS
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_DATA_DIR = BASE_DIR / "training_data"
FINETUNE_OUTPUT   = BASE_DIR / "finetune_output"

_finetune_running  = False
_finetune_lock     = threading.Lock()
_finetune_proc     = None
_finetune_log      = []   # live log lines from the subprocess


def _ft_init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS training_data (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_fn    TEXT,
        transcript  TEXT,
        uploaded_at TEXT,
        duration_s  REAL,
        used_in_run INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS finetune_runs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at   TEXT,
        finished_at  TEXT,
        status       TEXT DEFAULT 'running',
        samples_used INTEGER DEFAULT 0,
        last_loss    REAL,
        output_dir   TEXT,
        error_msg    TEXT
    )""")
    conn.commit()
    conn.close()


_ft_init_db()


def _ft_new_run(n_samples):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO finetune_runs (started_at, status, samples_used, output_dir) VALUES (?,?,?,?)",
              (datetime.now().isoformat(), "running", n_samples, str(FINETUNE_OUTPUT)))
    run_id = c.lastrowid
    conn.commit()
    conn.close()
    return run_id


def _ft_finish_run(run_id, ok, last_loss=None, error_msg=None):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""UPDATE finetune_runs
                    SET status=?, finished_at=?, last_loss=?, error_msg=?
                    WHERE id=?""",
                 ("done" if ok else "failed",
                  datetime.now().isoformat(),
                  last_loss, error_msg, run_id))
    conn.commit()
    conn.close()


def _ft_count_all():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM training_data")
    n = c.fetchone()[0]
    conn.close()
    return n


def _is_finetune_window():
    h       = datetime.now().hour
    start_h = int(os.environ.get("FINETUNE_START_HOUR", "22"))
    end_h   = int(os.environ.get("FINETUNE_END_HOUR",   "6"))
    if start_h > end_h:
        return h >= start_h or h < end_h
    return start_h <= h < end_h


def _start_finetune():
    global _finetune_running, _finetune_proc, _finetune_log
    with _finetune_lock:
        if _finetune_running:
            return None
        n = _ft_count_all()
        if n == 0:
            return None
        run_id = _ft_new_run(n)
        _finetune_log = [f"Finomhangolás indul (run #{run_id}, {n} minta összesen)..."]
        _finetune_running = True

    def _worker():
        global _finetune_running, _finetune_proc, _finetune_log
        last_loss = None
        proc = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "finetune_run.py"), "--run-id", str(run_id)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        _finetune_proc = proc
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            _finetune_log.append(line)
            if len(_finetune_log) > 500:
                _finetune_log = _finetune_log[-500:]
            m = re.search(r"loss=([\d.]+)", line)
            if m:
                last_loss = float(m.group(1))
        proc.stdout.close()
        rc = proc.wait()
        _finetune_proc = None
        ok = rc == 0
        _ft_finish_run(run_id, ok, last_loss=last_loss,
                       error_msg=None if ok else f"Kilépési kód: {rc}")
        _finetune_log.append("=== Kész ===" if ok else f"=== HIBA (kód {rc}) ===")
        with _finetune_lock:
            _finetune_running = False

    threading.Thread(target=_worker, daemon=True).start()
    return run_id


def _finetune_scheduler():
    """Automatikus indítás – csak ha FINETUNE_AUTO=1 van a .env-ben (alapból ki van kapcsolva)."""
    time.sleep(120)
    while True:
        time.sleep(600)
        if os.environ.get("FINETUNE_AUTO", "0") != "1":
            continue
        if not _is_finetune_window():
            continue
        if _finetune_running or _worker_running:
            continue
        if _ft_count_all() > 0:
            _start_finetune()


threading.Thread(target=_finetune_scheduler, daemon=True).start()


# ── Finomhangolás – lapok és API ──────────────────────────────────────────────

@app.route("/finetune")
def finetune_page():
    return render_template("finetune.html")


def _parse_paragraphs(text):
    parts = re.split(r'\n\s*\n', text.strip())
    return [re.sub(r'\s+', ' ', p).strip() for p in parts if p.strip()]


def _read_mp3_marker_cuts_ms(filepath):
    """
    Adobe Audition XMP marker-ek beolvasása PRIV:XMP ID3 frame-ből.
    Az Audition a markereket xmpDM:startTime értékként tárolja f48000-es
    idoalapban (48000 = sample rate, osztó az ms-re alakításhoz).
    Visszatér az összes marker ms pozíciójával (növekvo sorrend).
    """
    try:
        from mutagen.id3 import ID3
        tags = ID3(str(filepath))

        # ── 1. Standard CHAP tag-ek (ha mégis ezt írná) ───────────────────
        chaps = tags.getall('CHAP')
        if chaps:
            starts = sorted(ch.start_time for ch in chaps)
            return [t for t in starts if t > 50]

        # ── 2. PRIV frame amiben XMP adat van (Adobe Audition CC+) ────────
        xmp_str = None
        for key in tags.keys():
            if key.startswith('PRIV'):
                data = tags[key].data
                if b'xmpDM:startTime' in data:
                    xmp_str = data.decode('utf-8', errors='ignore')
                    break

        if not xmp_str:
            return []

        # Frame rate: "f48000" -> 48000
        fr_match = re.search(r'xmpDM:frameRate="f(\d+)"', xmp_str)
        frame_rate = int(fr_match.group(1)) if fr_match else 48000

        # Csak a CuePoint Markers track startTime értékei
        cue_block = re.search(
            r'xmpDM:trackName="CuePoint Markers"(.+?)xmpDM:trackName=',
            xmp_str, re.DOTALL
        )
        block = cue_block.group(1) if cue_block else xmp_str
        raw_times = [int(m) for m in re.findall(r'xmpDM:startTime="(\d+)"', block)]

        if not raw_times:
            return []

        cuts_ms = sorted(int(t / frame_rate * 1000) for t in raw_times)
        print(f"[MARKER] {len(cuts_ms)} db XMP marker (f{frame_rate}): {cuts_ms} ms", flush=True)
        return cuts_ms

    except Exception as e:
        print(f"[MARKER] Hiba: {e}", flush=True)
        return []


def _split_seg_by_cuts(seg, cut_points_ms, paragraphs):
    """
    Audio darabolása marker pozíciók szerint, bekezdések párosításával.

    N marker (elválasztó) -> N+1 szegmens -> N+1 bekezdéssel párosítva.
    Ha a szegmensek és bekezdések száma eltér, figyelmeztetést ír ki
    és amennyit tud, annyit párosít.
    """
    total  = len(seg)
    cuts   = [c for c in sorted(cut_points_ms) if 0 < c < total]
    bounds = [0] + cuts + [total]
    segs   = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    n_p = len(paragraphs)
    n_s = len(segs)

    if n_s != n_p:
        print(f"[MARKER] FIGYELEM: {n_s} szegmens vs {n_p} bekezdés – "
              f"elso {min(n_s, n_p)} db párosítva", flush=True)

    return [(seg[s:e], paragraphs[i]) for i, (s, e) in enumerate(segs) if i < n_p]


def _split_seg_by_silence(seg, paragraphs):
    """Fallback: N bekezdéshez az N-1 leghosszabb csend alapján vág."""
    from pydub.silence import detect_silence as _ds
    total = len(seg)
    n = len(paragraphs)
    if n == 1:
        return [(seg, paragraphs[0])]
    silences = _ds(seg, min_silence_len=200, silence_thresh=-40)
    if len(silences) >= n - 1:
        best = sorted(silences, key=lambda s: s[1] - s[0], reverse=True)[:n - 1]
        cuts = sorted([(s + e) // 2 for s, e in best])
    else:
        step = total // n
        cuts = [step * i for i in range(1, n)]
    bounds = [0] + cuts + [total]
    return [(seg[bounds[i]:bounds[i + 1]], paragraphs[i]) for i in range(n)]


def _save_training_sample(audio_fn, transcript, duration_s):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO training_data (audio_fn, transcript, uploaded_at, duration_s) VALUES (?,?,?,?)",
              (audio_fn, transcript, datetime.now().isoformat(), duration_s))
    conn.commit()
    conn.close()


@app.route("/finetune/upload", methods=["POST"])
def finetune_upload():
    audio = request.files.get("audio")
    transcript = request.form.get("transcript", "").strip()
    if not audio or not audio.filename:
        return jsonify({"error": "Nincs hangfájl!"})
    if not transcript:
        return jsonify({"error": "A leirat szövege kötelező!"})

    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    prefix    = datetime.now().strftime("%Y%m%d%H%M%S%f")[:18]
    orig_fn   = f"{prefix}_{audio.filename}"
    save_path = TRAINING_DATA_DIR / orig_fn
    audio.save(save_path)

    try:
        seg        = AudioSegment.from_file(str(save_path)).set_frame_rate(16000).set_channels(1)
        duration_s = len(seg) / 1000.0
        stem       = orig_fn.rsplit(".", 1)[0]

        if duration_s > 35:
            # Hosszú fájl darabolása
            paragraphs = _parse_paragraphs(transcript)
            if len(paragraphs) < 2:
                save_path.unlink(missing_ok=True)
                return jsonify({"error": (
                    f"A hanganyag túl hosszú ({duration_s:.1f} s). "
                    "Maximum 30 mp engedélyezett. Hosszabb fájlhoz a leiratban "
                    "minden bekezdés (üres sorral elválasztva) = egy hírelem."
                )})

            # 1. prioritás: Adobe Audition marker-ek (CHAP ID3 tag)
            marker_cuts = _read_mp3_marker_cuts_ms(save_path)
            save_path.unlink(missing_ok=True)

            if marker_cuts:
                split_method = "marker"
                chunks = _split_seg_by_cuts(seg, marker_cuts, paragraphs)
            else:
                split_method = "silence"
                chunks = _split_seg_by_silence(seg, paragraphs)

            saved = 0
            for i, (chunk_seg, chunk_text) in enumerate(chunks, 1):
                dur = len(chunk_seg) / 1000.0
                if dur < 1.5 or not chunk_text:
                    continue
                pfx  = datetime.now().strftime("%Y%m%d%H%M%S%f")[:18]
                cfn  = f"{pfx}_{i:03d}_{Path(audio.filename).stem}.wav"
                chunk_seg.export(str(TRAINING_DATA_DIR / cfn), format="wav")
                _save_training_sample(cfn, chunk_text, dur)
                saved += 1
            return jsonify({"ok": True, "split": True, "chunks": saved,
                            "split_method": split_method,
                            "duration_s": round(duration_s, 1)})

        # Normál rövid fájl: WAV-ra konvertálás
        wav_fn   = stem + ".wav"
        wav_path = TRAINING_DATA_DIR / wav_fn
        seg.export(str(wav_path), format="wav")
        if orig_fn != wav_fn:
            save_path.unlink(missing_ok=True)
    except Exception as e:
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({"error": f"Hangfájl feldolgozási hiba: {e}"})

    _save_training_sample(wav_fn, transcript, duration_s)
    return jsonify({"ok": True, "duration_s": round(duration_s, 1)})


@app.route("/finetune/data")
def finetune_data():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""SELECT id, audio_fn, transcript, uploaded_at, duration_s, used_in_run
                 FROM training_data ORDER BY id DESC""")
    rows = c.fetchall()
    conn.close()
    return jsonify({"items": [{
        "id": r[0], "audio_fn": r[1],
        "transcript": r[2], "uploaded_at": r[3],
        "duration_s": round(r[4] or 0, 1),
        "used": bool(r[5]),
    } for r in rows]})


@app.route("/finetune/audio/<path:filename>")
def finetune_audio(filename):
    return send_from_directory(str(TRAINING_DATA_DIR), filename)


@app.route("/finetune/data/<int:item_id>/delete", methods=["POST"])
def finetune_delete(item_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT audio_fn FROM training_data WHERE id=?", (item_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Nem található"})
    c.execute("DELETE FROM training_data WHERE id=?", (item_id,))
    conn.commit()
    conn.close()
    try:
        (TRAINING_DATA_DIR / row[0]).unlink(missing_ok=True)  # row[0] = audio_fn
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/finetune/status")
def finetune_status():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""SELECT id, started_at, finished_at, status, samples_used, last_loss, error_msg
                 FROM finetune_runs ORDER BY id DESC LIMIT 10""")
    runs = [{"id": r[0], "started_at": r[1], "finished_at": r[2],
             "status": r[3], "samples_used": r[4],
             "last_loss": r[5], "error_msg": r[6]} for r in c.fetchall()]
    total = _ft_count_all()
    conn.close()

    active_dir    = os.environ.get("WHISPER_MODEL_DIR", "").strip()
    ft_ready      = (FINETUNE_OUTPUT / "config.json").exists()
    active_is_ft  = bool(active_dir and Path(active_dir).resolve() == FINETUNE_OUTPUT.resolve())
    model_id      = os.environ.get("WHISPER_MODEL_ID", "Trendency/whisper-large-v3-hu")

    return jsonify({
        "running":          _finetune_running,
        "queue_busy":       _worker_running,
        "total_samples":    total,
        "model_ready":      ft_ready,
        "model_dir":        str(FINETUNE_OUTPUT),
        "active_model":     "finetune" if active_is_ft else "original",
        "active_label":     str(FINETUNE_OUTPUT) if active_is_ft else model_id,
        "log_tail":         _finetune_log[-80:],
        "runs":             runs,
    })


@app.route("/finetune/trigger", methods=["POST"])
def finetune_trigger():
    if _finetune_running:
        return jsonify({"error": "Már fut egy finomhangolás!"})
    if _worker_running:
        return jsonify({"error": "Leiratozás folyamatban – nem lehet egyszerre futtatni!"})
    if _ft_count_all() == 0:
        return jsonify({"error": "Nincs feltöltött tanítóadat!"})
    run_id = _start_finetune()
    return jsonify({"ok": True, "run_id": run_id})


@app.route("/finetune/stop", methods=["POST"])
def finetune_stop():
    global _finetune_running
    p = _finetune_proc
    if p and p.poll() is None:
        p.kill()
        _finetune_log.append("Manuálisan leállítva.")
        return jsonify({"ok": True})
    return jsonify({"error": "Nincs futó finomhangolás."})


@app.route("/finetune/activate", methods=["POST"])
def finetune_activate():
    if not (FINETUNE_OUTPUT / "config.json").exists():
        return jsonify({"error": "Nincs kész finomhangolt modell!"})
    set_key(str(ENV_FILE), "WHISPER_MODEL_DIR", str(FINETUNE_OUTPUT))
    os.environ["WHISPER_MODEL_DIR"] = str(FINETUNE_OUTPUT)
    return jsonify({"ok": True, "active": "finetune", "model_dir": str(FINETUNE_OUTPUT)})


@app.route("/finetune/revert", methods=["POST"])
def finetune_revert():
    set_key(str(ENV_FILE), "WHISPER_MODEL_DIR", "")
    os.environ["WHISPER_MODEL_DIR"] = ""
    model_id = os.environ.get("WHISPER_MODEL_ID", "Trendency/whisper-large-v3-hu")
    return jsonify({"ok": True, "active": "original", "model_id": model_id})


@app.route("/restart", methods=["POST"])
def restart_app():
    if _finetune_running:
        return jsonify({"error": "Finomhangolás fut – előbb állítsd le!"})
    def _do():
        time.sleep(0.8)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    threading.Thread(target=_do, daemon=False).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=PORT, debug=debug_mode, threaded=True)
