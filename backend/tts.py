# backend/tts.py
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

if not ELEVEN_KEY or not VOICE_ID:
    raise ValueError("ELEVENLABS_API_KEY or ELEVEN_VOICE_ID not set in server environment (.env)")

ELEVEN_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

def text_to_speech(text: str, out_path: str):
    headers = {
        "xi-api-key": ELEVEN_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/wav"
    }
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.6, "similarity_boost": 0.75}
    }
    resp = requests.post(ELEVEN_URL, headers=headers, json=payload, stream=True, timeout=60)
    resp.raise_for_status()
    # write stream to file
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    # small sleep to ensure file system sync
    time.sleep(0.05)
    return out_path
