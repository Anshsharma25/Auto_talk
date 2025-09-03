# Updated app.py — fixes for SadTalker invocation and safetensors handling
# - Uses sys.executable for subprocess so the same Python interpreter is used
# - Makes PYTHONPATH include SADTALKER_DIR so `-m sadtalker` can find the package
# - More robust search for an inference script inside SADTALKER_DIR (walks tree)
# - Handles both .safetensor and .safetensors extensions when trying to convert to .pth
# - If safetensors or torch are not installed, conversion is skipped with a warning (log preserved)
# - Writes an informative log file regardless of success/failure (already in original code)

import os
import uuid
import subprocess
import tempfile
import time
import sys
from flask import Flask, request, render_template, jsonify, send_from_directory
import requests
import pyttsx3
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import which as pydub_which

load_dotenv()

# -------------------------
# Ensure pydub can find ffmpeg
# -------------------------
_ffmpeg_path = r"C:\Users\HP\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
if os.path.exists(_ffmpeg_path):
    AudioSegment.converter = _ffmpeg_path
    os.environ["PATH"] += os.pathsep + os.path.dirname(_ffmpeg_path)
else:
    _ffmpeg_path = pydub_which("ffmpeg") or pydub_which("ffmpeg.exe")
    if _ffmpeg_path:
        AudioSegment.converter = _ffmpeg_path
    else:
        AudioSegment.converter = None
        print("WARNING: ffmpeg not found by pydub. Please check your installation path.")

# -------------------------
# Config (from .env)
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_URL = os.getenv("GROQ_CHAT_URL", "https://api.groq.com/openai/v1/chat/completions")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
SADTALKER_DIR = os.getenv("SADTALKER_DIR", "SadTalker")
SADTALKER_CHECKPOINT = os.getenv(
    "SADTALKER_CHECKPOINT",
    os.path.join(SADTALKER_DIR, "checkpoints", "sadtalker.pth"),
)
SADTALKER_DEVICE = os.getenv("SADTALKER_DEVICE", "cpu")

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() in ("1", "true", "yes")

os.makedirs(UPLOAD_DIR, exist_ok=True)
app = Flask(__name__, template_folder="templates")

# -------------------------
# Helpers
# -------------------------

def groq_chat_reply(user_text, conversation_history=None):
    if not GROQ_API_KEY:
        return f"(demo) I heard: {user_text}"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})
    payload = {
        "model": "gemma2-9b-it",
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.8,
    }
    resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    choice = data.get("choices", [{}])[0]
    text = choice.get("message", {}).get("content") or choice.get("text") or ""
    return text.strip()


def text_to_speech_wav(text, out_path_wav):
    if not out_path_wav.lower().endswith(".wav"):
        raise ValueError("out_path_wav must end with .wav")
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", 150)
    except Exception:
        pass
    engine.save_to_file(text, out_path_wav)
    engine.runAndWait()


def ensure_wav_16k_mono(src_wav, dst_wav=None):
    if dst_wav is None:
        dst_wav = src_wav
    if AudioSegment.converter is None:
        print("WARNING: pydub converter not configured; ensure ffmpeg is installed.")
    audio = AudioSegment.from_file(src_wav)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(dst_wav, format="wav")
    return dst_wav


def maybe_convert_safetensor(checkpoint_path):
    """
    If the checkpoint is a safetensors file, try to convert to a temporary .pth file
    Returns (actual_checkpoint_path, tmp_created_path_or_None).
    If safetensors/torch are not available, skip conversion with a warning.
    Handles both .safetensor and .safetensors endings.
    """
    lower = checkpoint_path.lower()
    if lower.endswith(".safetensor") or lower.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            import torch
        except Exception as e:
            # Don't crash here — log warning and fall back to passing original file
            print(f"WARNING: unable to import safetensors/torch to convert checkpoint: {e}")
            return checkpoint_path, None
        sd = load_file(checkpoint_path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        tmp_name = tmp.name
        tmp.close()
        torch.save(sd, tmp_name)
        return tmp_name, tmp_name
    return checkpoint_path, None


def find_infer_script():
    """Search the SADTALKER_DIR for a likely inference script. If found, return (path, False).
    If not found, return (None, True) meaning caller should attempt module execution (-m).
    """
    candidates = ("inference.py", "demo.py", "infer.py")
    # quick check in root
    for c in candidates:
        p = os.path.join(SADTALKER_DIR, c)
        if os.path.exists(p):
            return p, False
    # walk tree for any file with names containing inference/demo/infer
    for root, dirs, files in os.walk(SADTALKER_DIR):
        for f in files:
            lf = f.lower()
            if lf in candidates or any(tok in lf for tok in ("inference", "demo", "infer")):
                if f.endswith(".py"):
                    return os.path.join(root, f), False
    # not found — caller will try module form
    return None, True


def generate_sadtalker_video(
    image_path, audio_path, out_video_path, checkpoint_path=SADTALKER_CHECKPOINT, device="cpu", timeout=1800
):
    if not os.path.exists(SADTALKER_DIR):
        raise FileNotFoundError(f"SadTalker directory not found: {SADTALKER_DIR}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SadTalker checkpoint not found: {checkpoint_path}")

    actual_checkpoint, tmp_created = maybe_convert_safetensor(checkpoint_path)
    script_path, use_module = find_infer_script()

    abs_image = os.path.abspath(image_path)
    abs_audio = os.path.abspath(audio_path)
    abs_out = os.path.abspath(out_video_path)
    abs_checkpoint = os.path.abspath(actual_checkpoint)

    # Use the same Python interpreter that's running this app to avoid "wrong python" issues
    python_exec = sys.executable or "python"

    if not use_module and script_path:
        cmd = [python_exec, script_path, "--source_image", abs_image, "--driven_audio", abs_audio, "--output", abs_out, "--checkpoint", abs_checkpoint, "--device", device]
    else:
        # Try module invocation. Ensure SADTALKER_DIR is on PYTHONPATH so -m can find it
        cmd = [python_exec, "-m", "sadtalker", "--source_image", abs_image, "--driven_audio", abs_audio, "--output", abs_out, "--checkpoint", abs_checkpoint, "--device", device]

    env = os.environ.copy()
    # Prepend SADTALKER_DIR to PYTHONPATH so imports inside sadtalker package work
    old_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.path.abspath(SADTALKER_DIR) + (os.pathsep + old_py if old_py else "")

    proc = subprocess.run(
        cmd, cwd=SADTALKER_DIR if os.path.isdir(SADTALKER_DIR) else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")

    log_name = f"sadtalker_log_{int(time.time())}_{uuid.uuid4().hex}.txt"
    log_path = os.path.join(UPLOAD_DIR, log_name)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        fh.write("CHECKPOINT (used):\n" + abs_checkpoint + "\n\n")
        fh.write("STDOUT:\n" + stdout + "\n\n")
        fh.write("STDERR:\n" + stderr + "\n\n")

    if tmp_created:
        try:
            os.unlink(tmp_created)
        except Exception:
            pass

    if proc.returncode != 0:
        raise RuntimeError(f"SadTalker failed (rc={proc.returncode}). Log: {log_name}\nLast stderr:\n{stderr[-2000:]}")

    if not os.path.exists(abs_out):
        raise RuntimeError(f"SadTalker finished with rc=0 but output file missing. Log: {log_name}")

    return abs_out, log_name

# -------------------------
# Flask routes (unchanged)
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_image", methods=["POST"])
def upload_image():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no image file"}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)
    return jsonify({"image_path": filename})


@app.route("/chat", methods=["POST"])
def chat_and_animate():
    data = request.get_json(force=True)
    user_text = data.get("message", "")
    image_filename = data.get("image_filename", "")
    if not user_text or not image_filename:
        return jsonify({"error": "provide message and image_filename"}), 400

    try:
        reply_text = groq_chat_reply(user_text)
    except Exception as e:
        return jsonify({"error": "LLM error", "detail": str(e)}), 500

    uid = uuid.uuid4().hex
    audio_wav = os.path.join(UPLOAD_DIR, f"{uid}.wav")
    try:
        text_to_speech_wav(reply_text, audio_wav)
        ensure_wav_16k_mono(audio_wav)
    except Exception as e:
        return jsonify({"error": "TTS/resample failed", "detail": str(e)}), 500

    src_img_path = os.path.join(UPLOAD_DIR, image_filename)
    out_video = os.path.join(UPLOAD_DIR, f"{uid}_sadtalker_out.mp4")
    if not os.path.exists(src_img_path):
        return jsonify({"error": "image not found", "path": src_img_path}), 400

    try:
        generated_path, log_name = generate_sadtalker_video(
            src_img_path, audio_wav, out_video, checkpoint_path=SADTALKER_CHECKPOINT, device=SADTALKER_DEVICE
        )
    except Exception as e:
        return jsonify({"error": "SadTalker failed", "detail": str(e), "log_file": f"/uploads/{log_name}" if 'log_name' in locals() else None}), 500

    video_url = f"/uploads/{os.path.basename(generated_path)}"
    return jsonify({"reply_text": reply_text, "video_url": video_url, "log_file": f"/uploads/{log_name}"})


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
