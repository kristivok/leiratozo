import json
import os
import re
import subprocess
import sys

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")

PROMPT = """Feladat: a kapott leiratot ellenorizd es csak minimalisan javitsd.

KIMENETI FORMA:
- Csak a javitott leiratot add vissza, semmi mast.
- Nincs cim, nincs lista, nincs osszefoglalo.
- Ha nem tudod betartani a szabalyokat, add vissza a leiratot valtoztatas nelkul.

Fontos szabalyok:

TARTALMI VÁLTOZTATÁS TILOS.

Nem fogalmazhatsz át mondatokat.

Nem húzhatsz ki gondolatot, félmondatot, információt.

Nem tehetsz hozzá semmit, ami nincs a szövegben.

A mondatok sorrendje és szerkezete maradjon változatlan.

Megengedett javítások:

elgépelések, félrehallásból eredő nyilvánvaló hibák javítása

duplikált szavak és egymás után ismétlődő mondattöredékek egyszeri szerepeltetése

nevek, eseménynevek, fogalmak következetes és helyes írása

központozás, sortörés, beszélőváltások jelölése az olvashatóság érdekében

Stílus:

nyers rádiós leirat maradjon

nem irodalmi, nem szerkesztett, nem publicisztikai szöveg

élőbeszéd-jelleg megőrzése

Kimenet:

a teljes szöveg hiánytalanul

beszélők megjelölésével

egybefuggon, kihagyas nelkul"""

def build_text(data):
    segments = data.get("transcription", [])
    lines = []
    for seg in segments:
        speaker = str(seg.get("speaker", "")).strip()
        text = (seg.get("text") or "").strip()
        if not speaker:
            speaker = "SPEAKER"
        if text:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(f"{speaker}:")
    return "\n".join(lines).strip()

def is_valid_output(input_text, output_text):
    in_words = len(re.findall(r"\\S+", input_text))
    out_words = len(re.findall(r"\\S+", output_text))
    if out_words < max(10, int(in_words * 0.7)):
        return False
    in_speakers = sum(1 for line in input_text.splitlines() if ":" in line)
    out_speakers = sum(1 for line in output_text.splitlines() if ":" in line)
    if out_speakers < max(1, int(in_speakers * 0.7)):
        return False
    return True

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: llm_refine.py <input_json> <output_txt>\n")
        return 2
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    input_text = build_text(data)
    if not input_text:
        sys.stderr.write("Empty transcription input\n")
        return 3
    prompt = PROMPT + "\n\nLeirat:\n" + input_text + "\n\nJavitott leirat:\n"
    result = subprocess.run(
        ["ollama", "run", MODEL],
        input=prompt,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    output = (result.stdout or "").strip()
    if not output:
        sys.stderr.write("Empty model output\n")
        output = input_text
    if not is_valid_output(input_text, output):
        sys.stderr.write("Invalid model output, fallback to original\n")
        output = input_text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
