# app.py
import os
import json
import traceback
import uuid
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.utils import b64_to_bytes, bytes_to_b64, save_bytes_to_file, webm_to_wav
from backend.stt import transcribe_audio
from backend.llm import get_llm_response
from backend.tts import text_to_speech
from backend.avatar import generate_talking_video

BASE = Path.cwd()
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

FRONTEND_DIR = BASE / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/")
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/talk", methods=["POST"])
def talk():
    """
    Expect multipart/form-data:
      - image: file (jpg/png)  (required)
      - audio: file (webm/wav/ogg) (required)
    Returns JSON: {reply_text, audio_b64, video_b64}
    """
    try:
        if 'image' not in request.files or 'audio' not in request.files:
            return jsonify({"error": "image and audio files required"}), 400

        img_file = request.files['image']
        audio_file = request.files['audio']

        img_fn = secure_filename(img_file.filename or f"face_{uuid.uuid4().hex}.jpg")
        audio_fn = secure_filename(audio_file.filename or f"input_{uuid.uuid4().hex}.webm")

        face_path = DATA / f"face_{uuid.uuid4().hex[:8]}_{img_fn}"
        in_audio_path = DATA / f"input_{uuid.uuid4().hex[:8]}_{audio_fn}"

        img_file.save(str(face_path))
        audio_file.save(str(in_audio_path))

        # safer conversion: only convert if input not already WAV
        wav_path = DATA / f"input_{uuid.uuid4().hex[:8]}.wav"
        in_suffix = in_audio_path.suffix.lower()

        if in_suffix in [".wav", ".wave"]:
            wav_path = in_audio_path
        else:
            try:
                webm_to_wav(in_audio_path, wav_path)
            except RuntimeError as exc:
                # return helpful JSON with short excerpt of the error
                detail_lines = str(exc).splitlines()
                short_detail = "\n".join(detail_lines[:8])
                return jsonify({
                    "error": "ffmpeg missing or conversion failed",
                    "detail": short_detail
                }), 500

        # STT
        user_text = transcribe_audio(str(wav_path))
        if not isinstance(user_text, str):
            user_text = str(user_text or "")

        # LLM
        reply_text = get_llm_response(user_text)

        # TTS
        reply_audio_path = DATA / f"reply_{uuid.uuid4().hex[:8]}.wav"
        text_to_speech(reply_text, str(reply_audio_path))

        # Avatar (wav2lip)
        out_video_path = DATA / f"talk_{uuid.uuid4().hex[:8]}.mp4"
        generate_talking_video(str(face_path), str(reply_audio_path), str(out_video_path))

        # base64 encode outputs
        with open(reply_audio_path, "rb") as f:
            audio_b = f.read()
        with open(out_video_path, "rb") as f:
            video_b = f.read()

        resp = {
            "reply_text": reply_text,
            "audio_b64": bytes_to_b64(audio_b),
            "video_b64": bytes_to_b64(video_b)
        }

        # cleanup (best-effort)
        for p in (face_path, in_audio_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            if wav_path.exists() and wav_path != in_audio_path:
                wav_path.unlink(missing_ok=True)
        except Exception:
            pass
        for p in (reply_audio_path, out_video_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "server error", "detail": str(e)}), 500


if __name__ == "__main__":
    # bind to localhost so the page is served as a secure context for getUserMedia
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 8000)), debug=True)
