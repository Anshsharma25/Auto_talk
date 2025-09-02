# backend/utils.py
import base64
from pathlib import Path
import subprocess
import shutil
import os

def b64_to_bytes(b64: str) -> bytes:
    if ',' in b64:
        b64 = b64.split(',', 1)[1]
    return base64.b64decode(b64)

def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode('utf-8')

def save_bytes_to_file(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def _find_ffmpeg_executable():
    """
    Finds ffmpeg executable path. Checks:
      - FFMPEG_BINARY env var
      - system PATH via shutil.which("ffmpeg")
    Returns path to executable or None if not found.
    """
    env_path = os.environ.get("FFMPEG_BINARY")
    if env_path:
        # If it's just 'ffmpeg' this will also allow shutil.which to find it
        if shutil.which(env_path) or Path(env_path).exists():
            return env_path
    # fallback to system PATH
    ff = shutil.which("ffmpeg")
    return ff

def webm_to_wav(input_path, output_path, sample_rate=16000):
    """
    Convert webm/ogg/opus to wav via ffmpeg. Requires ffmpeg installed.
    Raises RuntimeError with actionable instructions if ffmpeg not found or if conversion fails.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    ffmpeg_exec = _find_ffmpeg_executable()
    if not ffmpeg_exec:
        raise RuntimeError(
            "ffmpeg executable not found. Install ffmpeg and ensure it's on your PATH.\n\n"
            "Windows quick options:\n"
            " - Install via Chocolatey (if you have it):  choco install ffmpeg -y\n"
            " - Install via Scoop (if you have it):        scoop install ffmpeg\n"
            " - Or download a build (eg from https://www.gyan.dev/ffmpeg/builds/) and add the 'bin' folder to your PATH.\n\n"
            "After installing, restart your terminal/IDE and run 'ffmpeg -version' or 'where ffmpeg' to confirm.\n"
            "If ffmpeg is installed at a custom path, set environment variable FFMPEG_BINARY to that full path."
        )

    cmd = [
        ffmpeg_exec, "-y", "-i", input_path,
        "-ar", str(sample_rate), "-ac", "1", output_path
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        msg = (
            f"ffmpeg conversion failed (returncode={e.returncode}).\n\n"
            f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}\n\n"
            "Check that input file is valid and ffmpeg can read it."
        )
        raise RuntimeError(msg)
    except FileNotFoundError:
        # defensive
        raise RuntimeError("ffmpeg not found when attempting to run it. Ensure ffmpeg is installed and on PATH.")
    return output_path
