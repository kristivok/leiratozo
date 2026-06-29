#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tanítóadat minőség-osztályozás Whisper finomhangoláshoz.

Cél: megjelölni, mely minták javíthatják a modellt és melyek vihetik félre a tanulást.
A hangsúly a „biztosan árt" eseteken van (audio↔szöveg eltérés, hallucináció, rossz hossz),
hogy egy túlreprezentált vagy hibás anyag ne tanítsa félre a modellt.

Két szint:
  - Heurisztikus (olcsó, GPU nélkül): szöveg + időtartam alapú jelek (cps, hossz, ismétlés…).
  - Modell-audit (GPU, opcionális): per-minta WER és loss – ezeket a training_audit.py tölti fel,
    és a classify() beépíti az osztályozásba.

A classify() egy minta címkéjét adja vissza:
  'bad'  ⛔ valószínűleg árt (kizárásra javasolt)
  'warn' ⚠ gyanús (nézd át)
  'ok'   ✅ valószínűleg hasznos
mindig indoklással.
"""
import re
from collections import Counter

# ── Küszöbök (tapasztalati értékek, magyar beszédre hangolva) ──────────────────
CPS_BAD_HIGH  = 30.0   # karakter/mp e fölött: a szöveg biztosan hosszabb, mint ami elhangozhat
CPS_WARN_HIGH = 22.0
CPS_WARN_LOW  = 5.0    # ez alatt gyanúsan kevés szöveg (csend / hiányzó szavak?)
CPS_BAD_LOW   = 2.0

DUR_MAX      = 29.0    # Whisper 30s-os ablaka – e fölött csonkolt tanítás
DUR_WARN_MAX = 27.0
DUR_MIN      = 1.0
DUR_WARN_MIN = 1.5

MIN_CHARS = 2

WER_BAD  = 0.60   # a modell átirata ennyire tér el a referenciától → valószínű hibás címke
WER_WARN = 0.40
LOSS_BAD  = 1.50  # magas per-minta loss (a tanítás ellenére is) → nehéz/hibás minta
LOSS_WARN = 1.00


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def norm_key(t: str) -> str:
    """Duplikátum-egyezéshez: kisbetűs, csak betű+szám, szóközök összevonva."""
    t = (t or "").lower()
    t = re.sub(r"[^0-9a-záéíóöőúüű ]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def detect_repetition(text: str):
    """Ismétlési hurok / hallucináció jele (str) vagy None.

    A transcript_after_diarization.clean_hallucination logikájával összhangban:
    karakter- és szószintű hurok, plusz egyetlen szó túlzott aránya.
    """
    t = text or ""
    if re.search(r"(.{2,8})\1{3,}", t):
        return "karakter-szintű ismétlési hurok (hallucináció?)"
    if re.search(r"\b(\w{1,4})\b(?:\s+\1\b){3,}", t, re.IGNORECASE):
        return "szó-szintű ismétlési hurok (hallucináció?)"
    words = re.findall(r"\w+", t.lower())
    if len(words) >= 8:
        w, c = Counter(words).most_common(1)[0]
        if c / len(words) > 0.4 and len(w) >= 3:
            return f"egy szó túlsúlya ('{w}' {c}×)"
    return None


def classify(text: str, duration_s, audit_wer=None, audit_loss=None) -> dict:
    """Egy minta osztályozása. Visszaad: {label, reasons[], cps}."""
    text = text or ""
    try:
        dur = float(duration_s or 0)
    except (TypeError, ValueError):
        dur = 0.0
    n   = len(_norm_text(text))
    cps = (n / dur) if dur > 0 else 0.0

    bad, warn = [], []

    # Üres / triviális leirat
    if n < MIN_CHARS:
        bad.append("üres vagy triviális leirat")

    # Időtartam
    if dur > DUR_MAX:
        bad.append(f"túl hosszú ({dur:.1f}s > {DUR_MAX:.0f}s, Whisper-ablakon túl → csonkolt)")
    elif dur >= DUR_WARN_MAX:
        warn.append(f"hosszú ({dur:.1f}s)")
    if 0 < dur < DUR_MIN:
        bad.append(f"túl rövid ({dur:.1f}s)")
    elif DUR_MIN <= dur < DUR_WARN_MIN:
        warn.append(f"rövid ({dur:.1f}s)")

    # Karakter/másodperc – audio↔szöveg illeszkedés (a legerősebb olcsó jel)
    if dur > 0 and n >= MIN_CHARS:
        if cps >= CPS_BAD_HIGH:
            bad.append(f"szöveg túl hosszú a hanghoz (cps={cps:.0f})")
        elif cps >= CPS_WARN_HIGH:
            warn.append(f"sűrű szöveg (cps={cps:.0f})")
        if dur >= 3 and cps <= CPS_BAD_LOW:
            bad.append(f"alig van szöveg a hanghoz (cps={cps:.1f})")
        elif dur >= 3 and cps <= CPS_WARN_LOW:
            warn.append(f"kevés szöveg a hanghoz (cps={cps:.1f})")

    # Ismétlés / hallucináció
    rep = detect_repetition(text)
    if rep:
        bad.append(rep)

    # Sok szám/írásjel (betűk aránya alacsony)
    if n >= 10:
        letters = sum(ch.isalpha() for ch in text)
        if letters / max(len(text), 1) < 0.55:
            warn.append("sok szám/írásjel a szövegben")

    # ── Modell-audit jelek (ha lefutott) ──────────────────────────────────────
    if audit_wer is not None:
        if audit_wer >= WER_BAD:
            bad.append(f"magas WER a modellhez képest ({audit_wer*100:.0f}%) → valószínű hibás címke")
        elif audit_wer >= WER_WARN:
            warn.append(f"emelt WER ({audit_wer*100:.0f}%)")
    if audit_loss is not None:
        if audit_loss >= LOSS_BAD:
            bad.append(f"magas tanítási loss ({audit_loss:.2f})")
        elif audit_loss >= LOSS_WARN:
            warn.append(f"emelt loss ({audit_loss:.2f})")

    label = "bad" if bad else ("warn" if warn else "ok")
    return {"label": label, "reasons": bad + warn, "cps": round(cps, 1)}


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (Levenshtein a szavakon) – egyszerű, függőség nélkül.

    Normalizálás: kisbetű, írásjelek el, szóközök össze. 0.0 = tökéletes egyezés.
    """
    ref = norm_key(reference).split()
    hyp = norm_key(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein távolság szólistákon
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(hyp)] / len(ref)
