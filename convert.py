import subprocess
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "audio.wav"

def convert_to_diarization_format(input_file, output_file=DEFAULT_OUTPUT):
    """
    Átkonvertálja a megadott bemeneti fájlt WAV formátumra,
    egy csatornás (mono) hanggal és 16 kHz-es mintavételi frekvenciával.
    """
    supported_formats = [
        '.mp4', '.mpg', '.mov', '.mxf',  # videó
        '.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a', '.wma'  # audió
    ]

    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext not in supported_formats:
        print(f"Hiba: Nem támogatott formátum! ({file_ext})")
        print(f"Támogatott formátumok: {', '.join(supported_formats)}")
        sys.exit(1)

    # ffmpeg parancs összeállítása
    command = [
        "ffmpeg",
        "-i", str(input_file),
        "-ac", "1",              # egy csatornás hang (mono)
        "-ar", "16000",           # 16 kHz-es mintavételi frekvencia
        "-acodec", "pcm_s16le",   # 16-bit PCM kódolás
        str(output_file),
        "-y"                      # felülírja a meglévő fájlt
    ]

    try:
        # A subprocess.run kimenetét elrejtjük, hogy ne zavarjon be a naplóba
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Hiba történt a konvertálás során: {e}")
        # A hiba részleteit a stderr-ből olvassuk ki
        print(f"FFmpeg hibaüzenet: {e.stderr}")
        sys.exit(1)

    if os.path.exists(output_file):
        print(f"Sikeres konvertálás! Az eredmény: {output_file}")
    else:
        print("Hiba: A kimeneti fájl nem jött létre!")
        sys.exit(1)

### --- MÓDOSÍTÁS --- ###
# A szkript most a standard bemenetről olvassa a fájlnevet,
# így az app.py át tudja adni neki.
if __name__ == "__main__":
    # Beolvassuk az első sort a standard inputról
    input_file_path = sys.stdin.readline().strip()

    if not input_file_path:
        print("Hiba: nem érkezett fájlnév a bemeneten.")
        sys.exit(1)

    input_path = Path(input_file_path)

    if not input_path.exists():
        print(f"Hiba: a megadott fájl nem található! ({input_file_path})")
        sys.exit(1)

    convert_to_diarization_format(input_path, DEFAULT_OUTPUT)
