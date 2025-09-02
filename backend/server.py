# backend/server.py
import os
import uuid
import base64
import traceback
from pathlib import Path
from flask import Flask, request, jsonify
from backend.utils import webm_to_wav
from backend.stt import transcribe_audio
from backend.llm import get_llm_response
from backend.tts import text_to_speech
from backend.avatar import generate_talking_video

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

def _b64_file_bytes(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@app.route("/talk", methods=["POST"])
def talk():
    try:
        if "image" not in request.files or "audio" not in request.files:
            return jsonify({"detail": "image and audio files required", "error": "bad request"}), 400

        uid = uuid.uuid4().hex[:8]
        img_f = request.files["image"]
        aud_f = request.files["audio"]

        face_path = DATA_DIR / f"face_{uid}{Path(img_f.filename).suffix or '.jpg'}"
        audio_raw = DATA_DIR / f"question_{uid}{Path(aud_f.filename).suffix or '.wav'}"

        img_f.save(str(face_path))
        aud_f.save(str(audio_raw))

        # convert audio to 16k mono wav (ffmpeg required)
        wav_path = DATA_DIR / f"question_{uid}.wav"
        try:
            webm_to_wav(audio_raw, wav_path, sample_rate=16000)
        except Exception:
            # if audio is already wav, fall back
            if audio_raw.suffix.lower() in [".wav", ".wave"]:
                wav_path = audio_raw
            else:
                raise

        # STT -> text
        user_text = transcribe_audio(str(wav_path))
        app.logger.info("User text: %s", user_text)

        # LLM -> reply
        reply_text = get_llm_response(user_text)
        app.logger.info("Reply text: %s", reply_text)

        # TTS -> reply wav
        reply_audio_path = DATA_DIR / f"reply_{uid}.wav"
        text_to_speech(reply_text, str(reply_audio_path))

        # Wav2Lip -> mp4
        out_video_path = DATA_DIR / f"talk_{uid}.mp4"
        generate_talking_video(str(face_path), str(reply_audio_path), str(out_video_path))

        audio_b64 = _b64_file_bytes(reply_audio_path)
        video_b64 = _b64_file_bytes(out_video_path)

        return jsonify({
            "reply_text": reply_text,
            "audio_b64": audio_b64,
            "video_b64": video_b64
        })

    except Exception as e:
        traceback_str = traceback.format_exc()
        app.logger.error("Error in /talk: %s\n%s", e, traceback_str)
        return jsonify({"detail": str(e), "error": "server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True)
