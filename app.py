# app.py
import os
import uuid
import subprocess
import tempfile
import time
from flask import Flask, request, render_template, jsonify, send_from_directory
import requests
import pyttsx3
from dotenv import load_dotenv

# pydub for audio handling/resampling
from pydub import AudioSegment
from pydub.utils import which as pydub_which

load_dotenv()

# -------------------------
# Ensure pydub can find ffmpeg
# -------------------------
# Explicitly set your ffmpeg path
_ffmpeg_path = r"C:\Users\HP\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

if os.path.exists(_ffmpeg_path):
    AudioSegment.converter = _ffmpeg_path
    os.environ["PATH"] += os.pathsep + os.path.dirname(_ffmpeg_path)
else:
    # fallback to PATH lookup
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
    """Produce a WAV file using pyttsx3 (offline)."""
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
    """Resample and convert audio to 16kHz mono WAV. Overwrites src by default."""
    if dst_wav is None:
        dst_wav = src_wav
    if AudioSegment.converter is None:
        print("WARNING: pydub converter not configured; ensure ffmpeg is installed.")
    audio = AudioSegment.from_file(src_wav)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(dst_wav, format="wav")
    return dst_wav


def maybe_convert_safetensor(checkpoint_path):
    if not checkpoint_path.lower().endswith(".safetensor"):
        return checkpoint_path, None
    from safetensors.torch import load_file
    import torch
    sd = load_file(checkpoint_path)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    tmp_name = tmp.name
    tmp.close()
    torch.save(sd, tmp_name)
    return tmp_name, tmp_name


def find_infer_script():
    candidates = ["inference.py", "demo.py", "infer.py"]
    for c in candidates:
        p = os.path.join(SADTALKER_DIR, c)
        if os.path.exists(p):
            return p, False
    return None, True


def generate_sadtalker_video(
    image_path, audio_path, out_video_path, checkpoint_path=SADTALKER_CHECKPOINT, device="cpu", timeout=900
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

    if not use_module:
        cmd = [
            "python",
            script_path,
            "--source_image",
            abs_image,
            "--driven_audio",
            abs_audio,
            "--output",
            abs_out,
            "--checkpoint",
            abs_checkpoint,
            "--device",
            device,
        ]
    else:
        cmd = [
            "python",
            "-m",
            "sadtalker",
            "--source_image",
            abs_image,
            "--driven_audio",
            abs_audio,
            "--output",
            abs_out,
            "--checkpoint",
            abs_checkpoint,
            "--device",
            device,
        ]

    proc = subprocess.run(
        cmd, cwd=SADTALKER_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy(), timeout=timeout
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
# Flask routes
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
        return jsonify({"error": "SadTalker failed", "detail": str(e)}), 500

    video_url = f"/uploads/{os.path.basename(generated_path)}"
    return jsonify({"reply_text": reply_text, "video_url": video_url, "log_file": f"/uploads/{log_name}"})


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
