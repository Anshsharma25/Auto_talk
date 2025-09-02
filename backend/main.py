# main.py
"""
Optional local conversation loop for testing (uses backend package functions).
Put an image at data/me.jpg before running.
"""
import uuid
from pathlib import Path
from backend.utils import webm_to_wav
from backend.stt import transcribe_audio
from backend.llm import get_llm_response
from backend.tts import text_to_speech
from backend.avatar import generate_talking_video

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INPUT_IMAGE = DATA_DIR / "me.jpg"   # Put your avatar face here
CHUNK_SECONDS = 3

def run_once_with_file(audio_input_path: Path):
    # convert if needed
    wav = DATA_DIR / f"tmp_{uuid.uuid4().hex[:8]}.wav"
    try:
        webm_to_wav(audio_input_path, wav)
    except Exception:
        if audio_input_path.suffix.lower() in [".wav", ".wave"]:
            wav = audio_input_path
        else:
            raise
    user_text = transcribe_audio(str(wav))
    print("User:", user_text)
    reply = get_llm_response(user_text)
    print("Agent:", reply)
    reply_audio = DATA_DIR / f"reply_{uuid.uuid4().hex[:8]}.wav"
    text_to_speech(reply, str(reply_audio))
    out_video = DATA_DIR / f"talk_{uuid.uuid4().hex[:8]}.mp4"
    generate_talking_video(str(INPUT_IMAGE), str(reply_audio), str(out_video))
    print("Generated:", out_video)
