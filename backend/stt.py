# backend/stt.py
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in server environment (.env)")

client = Groq(api_key=GROQ_API_KEY)

def transcribe_audio(path: str) -> str:
    """
    Sends an audio file to Groq Whisper (or whichever audio model) for transcription.
    Expects server-side environment key present.
    """
    with open(path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f
        )
    if hasattr(transcription, "text"):
        return transcription.text
    if isinstance(transcription, dict):
        return transcription.get("text", "") or transcription.get("transcription", "")
    return str(transcription)
