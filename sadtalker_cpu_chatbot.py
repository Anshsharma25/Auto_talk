"""
Realtime avatar chatbot with HeyGen integration (photo -> talking avatar video).

This version adds more robust handling of HeyGen avatar-creation APIs:
 - Detects and uses the image_key returned by upload if required.
 - Tries multiple payload shapes for adding the image to the avatar group.
 - If create-avatar-group without image_key fails, retries create including image_key.
 - Prints full responses to help debug API mismatches.
"""
import os
import sys
import time
import queue
import tempfile
import threading
import wave
import json
import argparse
import webbrowser
import mimetypes

import numpy as np
import cv2
import pyttsx3
import sounddevice as sd
import simpleaudio as sa
import requests
from dotenv import load_dotenv

# try tkinter for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

load_dotenv()

# ---------- HEYGEN API KEY ----------
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", None)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", os.path.join(os.getcwd(), "models", "vosk-model"))

def mask_key(k):
    if not k:
        return "<none>"
    k = k.strip()
    if len(k) <= 8:
        return "*" * (len(k) - 2) + k[-2:]
    return f"{k[:4]}...{k[-4:]}"

def test_heygen_key_once(key, timeout=10):
    if not key:
        return False, None, "HEYGEN_API_KEY not provided"
    try:
        url = "https://api.heygen.com/v2/avatars"
        r = requests.get(url, headers={"X-Api-Key": key, "Accept": "application/json"}, timeout=timeout)
        try:
            body = r.json()
        except Exception:
            body = r.text
        ok = (r.status_code == 200)
        return ok, r.status_code, body
    except Exception as e:
        return False, None, str(e)

# --- Groq (optional) ---
USE_GROQ = False
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        GROQ_MODEL = "gemma2-9b-it"
        USE_GROQ = True
        print("[INFO] Groq client ready. Using model:", GROQ_MODEL)
    except Exception as e:
        print("[WARN] Could not initialize Groq client:", e)
        USE_GROQ = False
else:
    print("[INFO] GROQ_API_KEY not set — Groq disabled, falling back to simple replies.")

# --- Vosk ASR
try:
    from vosk import Model, KaldiRecognizer
except Exception as e:
    print("ERROR: Vosk not installed. Install with: pip install vosk")
    raise

# --- Mediapipe (optional preview)
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
except Exception as e:
    mp_face = None
    face_mesh = None
    print("[WARN] mediapipe not available — local animation disabled.", e)

# ---------------------------
# HeyGen helpers (robust)
# ---------------------------
HEYGEN_UPLOAD_URL = "https://upload.heygen.com/v1/asset"
HEYGEN_API_BASE = "https://api.heygen.com"

def heygen_headers(content_type=None):
    if not HEYGEN_API_KEY:
        raise RuntimeError("HEYGEN_API_KEY not set. Put it in .env or export it.")
    headers = {"X-Api-Key": HEYGEN_API_KEY, "Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers

def upload_asset_to_heygen(image_path):
    """Upload raw bytes with Content-Type set to the MIME type. Return JSON or None."""
    print("[HeyGen] Uploading asset:", image_path)
    if not os.path.exists(image_path):
        print("[HeyGen] File not found:", image_path)
        return None
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        headers = heygen_headers(content_type=mime_type)
        resp = requests.post(HEYGEN_UPLOAD_URL, headers=headers, data=data, timeout=120)
        try:
            j = resp.json()
        except Exception:
            j = {"status_code": resp.status_code, "text": resp.text}
        print("[HeyGen] upload HTTP status:", resp.status_code)
        print("[HeyGen] upload response:", json.dumps(j) if isinstance(j, (dict,list)) else str(j))
        if resp.status_code not in (200,201):
            return None
        return j
    except Exception as e:
        print("[HeyGen] upload exception:", e)
        return None

def create_photo_avatar_group(name="avatar_group_from_api", image_key_included=None):
    """
    Tries to create avatar group; if image_key_included provided, include it in payload.
    Returns parsed JSON or None.
    """
    url = HEYGEN_API_BASE + "/v2/photo_avatar/avatar_group/create"
    payload = {"name": name}
    if image_key_included:
        # include key under a likely parameter name
        payload["image_key"] = image_key_included
    try:
        r = requests.post(url, headers=heygen_headers(), json=payload, timeout=30)
        try:
            j = r.json()
        except Exception:
            j = {"status_code": r.status_code, "text": r.text}
        print("[HeyGen] create_avatar_group status:", r.status_code)
        print("[HeyGen] create_avatar_group response:", json.dumps(j) if isinstance(j,(dict,list)) else str(j))
        if r.status_code not in (200,201):
            return None, j
        return j, None
    except Exception as e:
        print("[HeyGen] create_avatar_group exception:", e)
        return None, {"exception": str(e)}

def add_look_to_group(group_id, asset_key_or_url, look_name="look1"):
    """
    Try different payload shapes for adding the look. Returns JSON or None.
    """
    url = HEYGEN_API_BASE + "/v2/photo_avatar/avatar_group/add"
    # try several payload variants in order
    candidates = [
        {"avatar_group_id": group_id, "image_asset_key": asset_key_or_url, "name": look_name},
        {"avatar_group_id": group_id, "image_key": asset_key_or_url, "name": look_name},
        {"avatar_group_id": group_id, "image_asset_url": asset_key_or_url, "name": look_name},
        {"avatar_group_id": group_id, "image_url": asset_key_or_url, "name": look_name},
    ]
    for payload in candidates:
        try:
            r = requests.post(url, headers=heygen_headers(), json=payload, timeout=60)
            try:
                j = r.json()
            except Exception:
                j = {"status_code": r.status_code, "text": r.text}
            print("[HeyGen] add_look attempt payload keys:", list(payload.keys()))
            print("[HeyGen] add_look status:", r.status_code)
            print("[HeyGen] add_look response:", json.dumps(j) if isinstance(j,(dict,list)) else str(j))
            if r.status_code in (200,201):
                return j
            # if 400 but message suggests wrong field, continue to next candidate
        except Exception as e:
            print("[HeyGen] add_look exception for payload", payload.keys(), "->", e)
            continue
    return None

def train_photo_avatar_group(group_id):
    url = HEYGEN_API_BASE + "/v2/photo_avatar/train"
    payload = {"avatar_group_id": group_id}
    try:
        r = requests.post(url, headers=heygen_headers(), json=payload, timeout=30)
        try:
            j = r.json()
        except Exception:
            j = {"status_code": r.status_code, "text": r.text}
        print("[HeyGen] train status:", r.status_code)
        print("[HeyGen] train response:", json.dumps(j) if isinstance(j,(dict,list)) else str(j))
        if r.status_code not in (200,201):
            return None
        return j
    except Exception as e:
        print("[HeyGen] train exception:", e)
        return None

def check_training_status(job_id):
    url = HEYGEN_API_BASE + f"/v2/photo_avatar/generation/{job_id}"
    try:
        r = requests.get(url, headers=heygen_headers(), timeout=20)
        try:
            j = r.json()
        except Exception:
            j = {"status_code": r.status_code, "text": r.text}
        print("[HeyGen] check_training_status:", r.status_code, j if isinstance(j,(dict,list)) else str(j))
        if r.status_code != 200:
            return None
        return j
    except Exception as e:
        print("[HeyGen] training status exception:", e)
        return None

def list_talking_photos():
    url = HEYGEN_API_BASE + "/v2/avatars"
    try:
        r = requests.get(url, headers=heygen_headers(), timeout=20)
        try:
            j = r.json()
        except Exception:
            j = {"status_code": r.status_code, "text": r.text}
        print("[HeyGen] list_avatars status:", r.status_code)
        print("[HeyGen] list_avatars response:", json.dumps(j) if isinstance(j,(dict,list)) else str(j))
        if r.status_code != 200:
            return None
        return j
    except Exception as e:
        print("[HeyGen] list avatars exception:", e)
        return None

def create_avatar_video_with_talking_photo(talking_photo_id, text, voice_id=None, callback_url=None):
    url = HEYGEN_API_BASE + "/v2/video/generate"
    voice = {"type": "text", "input_text": text}
    if voice_id:
        voice["voice_id"] = voice_id
    video_inputs = [
        {
            "character": {
                "type": "talking_photo",
                "talking_photo_id": talking_photo_id,
                "talking_photo_style": "expressive"
            },
            "voice": voice,
            "background": {"type": "color", "value": "#f6f6fc"}
        }
    ]
    payload = {"video_inputs": video_inputs}
    if callback_url:
        payload["callback_url"] = callback_url
    try:
        r = requests.post(url, headers=heygen_headers(), json=payload, timeout=60)
        try:
            j = r.json()
        except Exception:
            j = {"status_code": r.status_code, "text": r.text}
        print("[HeyGen] video generate status:", r.status_code)
        print("[HeyGen] video generate response:", json.dumps(j) if isinstance(j,(dict,list)) else str(j))
        if r.status_code not in (200,201):
            return None
        vid = None
        if isinstance(j, dict):
            if "data" in j and isinstance(j["data"], dict):
                vid = j["data"].get("video_id") or j["data"].get("id")
            elif "video_id" in j:
                vid = j["video_id"]
        return vid
    except Exception as e:
        print("[HeyGen] video generate exception:", e)
        return None

def check_video_status(video_id):
    url = HEYGEN_API_BASE + f"/v1/video_status.get?video_id={video_id}"
    try:
        r = requests.get(url, headers=heygen_headers(), timeout=20)
        try:
            j = r.json()
        except Exception:
            j = {"status_code": r.status_code, "text": r.text}
        print("[HeyGen] check_video_status:", r.status_code, j if isinstance(j,(dict,list)) else str(j))
        if r.status_code != 200:
            return None
        return j
    except Exception as e:
        print("[HeyGen] video status exception:", e)
        return None

def get_video_url_from_status(status_json):
    try:
        data = status_json.get("data", {}) if isinstance(status_json, dict) else {}
        for k in ("video_url", "file_url", "video_url_https", "url"):
            if k in data and data[k]:
                return data[k]
        if "output" in data and isinstance(data["output"], dict):
            out = data["output"]
            for k in ("video_url", "file_url"):
                if k in out and out[k]:
                    return out[k]
    except Exception:
        pass
    return None

# ---------------------------
# Chat fallback reply
# ---------------------------
def generate_reply_with_groq(user_text, history=None, max_tokens=256):
    if not USE_GROQ:
        return None
    try:
        messages = [{"role": "user", "content": user_text}]
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content
        if isinstance(text, str):
            return text.strip()
        return str(text).strip()
    except Exception as e:
        print("[WARN] Groq call failed:", e)
        return None

def generate_reply(user_text, history=None):
    if USE_GROQ:
        out = generate_reply_with_groq(user_text, history)
        if out:
            return out
    u = user_text.lower()
    if any(x in u for x in ("hello", "hi", "hey")):
        return "Hi! I heard you. Ask me anything about the image."
    if "who" in u and "you" in u:
        return "I'm your friendly avatar created from your photo."
    if "age" in u:
        return "I can't determine exact ages, but you look great!"
    if "smile" in u or "happy" in u:
        return "Sure — I'll smile for you."
    return "Thanks for telling me. I'll do my best to answer."

# ---------------------------
# Vosk listener (unchanged)
# ---------------------------
class VoskListener:
    def __init__(self, model_path, samplerate=16000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at: {model_path}")
        self.model = Model(model_path)
        self.samplerate = samplerate
        self.rec = None
        self.q = queue.Queue()
        self.running = False
        self.thread = None
        self.result_callback = None

    def audio_callback(self, indata, frames, time_info, status):
        if self.rec is None:
            return
        try:
            data_bytes = indata.tobytes() if isinstance(indata, np.ndarray) else bytes(indata)
            if self.rec.AcceptWaveform(data_bytes):
                res = self.rec.Result()
                try:
                    j = json.loads(res)
                    text = j.get("text", "").strip()
                    if text:
                        self.q.put(text)
                except Exception:
                    pass
        except Exception as e:
            print("[VOSK] audio_callback error:", e)

    def start(self, result_callback):
        self.result_callback = result_callback
        self.rec = KaldiRecognizer(self.model, self.samplerate)
        self.running = True
        self.thread = threading.Thread(target=self._run_stream, daemon=True)
        self.thread.start()

    def _run_stream(self):
        try:
            with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000, dtype="int16",
                                   channels=1, callback=self.audio_callback):
                print("[VOSK] Listening... (speak now)")
                while self.running:
                    try:
                        text = self.q.get(timeout=0.5)
                        if self.result_callback:
                            try:
                                self.result_callback(text)
                            except Exception as e:
                                print("[VOSK] result_callback error:", e)
                    except queue.Empty:
                        continue
        except Exception as e:
            print("[VOSK] _run_stream error (input device?):", e)

    def stop(self):
        self.running = False

# ---------------------------
# Image handling helpers
# ---------------------------
def detect_face_landmarks(image_bgr):
    if face_mesh is None:
        return None
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results or not results.multi_face_landmarks:
        return None
    face = results.multi_face_landmarks[0]
    h, w = image_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]
    return np.array(pts, dtype=np.int32)

def load_avatar_image(path, target_size=(640,480)):
    if not path or not os.path.exists(path):
        print("[ERROR] File not found or invalid path:", path)
        return None, None
    img = cv2.imread(path)
    if img is None:
        print("[ERROR] Failed to load image (cv2.imread returned None):", path)
        return None, None
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    pts = None
    if face_mesh is not None:
        try:
            pts = detect_face_landmarks(img)
            if pts is None:
                print("[WARN] No face detected in uploaded image — local preview disabled.")
            else:
                print("[INFO] Local face landmarks detected (for preview).")
        except Exception as e:
            print("[WARN] detect_face_landmarks error:", e)
            pts = None
    return img, pts

def ensure_local_image(path_or_url):
    if not path_or_url:
        return None
    path_or_url = path_or_url.strip().strip('"').strip("'")
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        try:
            r = requests.get(path_or_url, timeout=30)
            if r.status_code == 200:
                ext = os.path.splitext(path_or_url)[1]
                if not ext or len(ext) > 8:
                    ext = ".jpg"
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(r.content)
                tmp.close()
                print("[INFO] downloaded remote image to", tmp.name)
                return tmp.name
            else:
                print("[WARN] failed to download image, status:", r.status_code)
                return None
        except Exception as e:
            print("[WARN] download failed:", e)
            return None
    else:
        return path_or_url

# ---------------------------
# Robust HeyGen avatar creation flow
# ---------------------------
def heygen_create_or_get_talking_photo(image_path, group_cache):
    key = os.path.abspath(image_path)
    if key in group_cache:
        return group_cache[key].get("talking_photo_id")

    up = upload_asset_to_heygen(image_path)
    if up is None:
        print("[HeyGen] asset upload failed.")
        return None

    # Extract best asset identifier from upload response
    asset_key = None
    try:
        if isinstance(up, dict):
            data = up.get("data") or up
            if isinstance(data, dict):
                asset_key = (data.get("image_key") or data.get("imageKey") or
                             data.get("asset_key") or data.get("key") or
                             data.get("id") or data.get("url") or data.get("file_url"))
    except Exception:
        asset_key = None

    # Fallback: if top-level contains image_key
    if not asset_key:
        try:
            asset_key = up.get("image_key") if isinstance(up, dict) else None
        except Exception:
            asset_key = None

    if not asset_key:
        print("[HeyGen] Warning: upload returned no usable asset key. Full upload response printed above.")
        # still continue: try creating group without asset key
    else:
        print("[HeyGen] Determined asset_key:", asset_key)

    # 1) Try creating avatar group without image_key first
    grp, err = create_photo_avatar_group(name=f"api_group_{int(time.time())}", image_key_included=None)
    group_id = None
    if grp and not err:
        # try to parse group id
        try:
            group_id = (grp.get("data", {}) or {}).get("avatar_group_id") or grp.get("avatar_group_id") or grp.get("id") or (grp.get("data", {}) or {}).get("id")
        except Exception:
            group_id = None

    # 2) If creation failed and asset_key exists, retry including image_key in payload (some APIs require it)
    if not group_id:
        if asset_key:
            print("[HeyGen] Retrying create_avatar_group including image_key (API variant).")
            grp2, err2 = create_photo_avatar_group(name=f"api_group_{int(time.time())}", image_key_included=asset_key)
            if grp2 and not err2:
                try:
                    group_id = (grp2.get("data", {}) or {}).get("avatar_group_id") or grp2.get("avatar_group_id") or grp2.get("id") or (grp2.get("data", {}) or {}).get("id")
                except Exception:
                    group_id = None
            else:
                print("[HeyGen] create with image_key also failed; response:", err2)
        else:
            print("[HeyGen] No asset_key to include when retrying create_avatar_group.")

    if not group_id:
        # If still no group_id, attempt to add the image directly via avatar_group/add by creating a group id placeholder? 
        # Some APIs allow creating avatar directly (not group). Try alternate endpoint '/v2/photo_avatar/create'
        if asset_key:
            alt_url = HEYGEN_API_BASE + "/v2/photo_avatar/create"
            payload = {"image_key": asset_key, "name": f"talking_photo_{int(time.time())}"}
            try:
                r = requests.post(alt_url, headers=heygen_headers(), json=payload, timeout=30)
                try:
                    j = r.json()
                except Exception:
                    j = {"status_code": r.status_code, "text": r.text}
                print("[HeyGen] alt create (photo_avatar/create) status:", r.status_code)
                print("[HeyGen] alt create response:", json.dumps(j) if isinstance(j,(dict,list)) else str(j))
                # alt response may include speaking/photo id directly
                if r.status_code in (200,201) and isinstance(j, dict):
                    talking_photo_id = j.get("data", {}).get("talking_photo_id") or j.get("talking_photo_id") or j.get("id")
                    if talking_photo_id:
                        print("[HeyGen] obtained talking_photo_id from alt create:", talking_photo_id)
                        group_cache[key] = {"group_id": None, "talking_photo_id": talking_photo_id}
                        return talking_photo_id
            except Exception as e:
                print("[HeyGen] alt create exception:", e)

    # If we have a group_id, try adding the look with various payload shapes
    if group_id:
        add_res = add_look_to_group(group_id, asset_key or image_path, look_name="initial_look")
        if not add_res:
            print("[HeyGen] add look failed for group:", group_id)
            return None
        # now train the group
        train_res = train_photo_avatar_group(group_id)
        if not train_res:
            print("[HeyGen] train request failed.")
            return None
        # parse job id or waiting id
        job_id = None
        try:
            if isinstance(train_res, dict):
                job_id = train_res.get("data", {}).get("job_id") or train_res.get("job_id") or train_res.get("id")
        except Exception:
            job_id = None
        if not job_id:
            job_id = group_id
        print("[HeyGen] Training started (job id):", job_id)
        timeout = 300
        interval = 5
        elapsed = 0
        talking_photo_id = None
        while elapsed < timeout:
            st = check_training_status(job_id)
            if not st:
                print("[HeyGen] training status request failed; will retry.")
            else:
                data = st.get("data") if isinstance(st, dict) else None
                print("[HeyGen] training status data:", data)
                if data:
                    talking_photo_id = data.get("talking_photo_id") or data.get("photo_id") or data.get("talking_photo") or data.get("photo_avatar_id")
                    if talking_photo_id:
                        print("[HeyGen] talking_photo_id found:", talking_photo_id)
                        break
                    status = data.get("status") or data.get("state")
                    if status and status.lower() in ("completed","succeeded","finished"):
                        break
                    if status and status.lower() in ("failed","error"):
                        print("[HeyGen] training failed:", st)
                        return None
            time.sleep(interval)
            elapsed += interval

        if not talking_photo_id:
            avs = list_talking_photos()
            if avs and isinstance(avs, dict):
                entries = avs.get("talking_photos") or avs.get("avatars") or (avs.get("data", {}) or {}).get("talking_photos")
                if entries and isinstance(entries, list) and len(entries) > 0:
                    tp = entries[0]
                    talking_photo_id = tp.get("talking_photo_id") or tp.get("id") or tp.get("avatar_id")
                    print("[HeyGen] chosen talking_photo_id from list:", talking_photo_id)
        if not talking_photo_id:
            print("[HeyGen] Could not obtain talking_photo_id automatically. Check HeyGen dashboard.")
            return None

        group_cache[key] = {"group_id": group_id, "talking_photo_id": talking_photo_id}
        return talking_photo_id

    # If we reach here, we couldn't make a talking photo
    print("[HeyGen] could not create or find talking photo for this image.")
    return None

# ---------------------------
# transcript handler factory
# ---------------------------
def on_transcript_factory(avatar_image_path, group_cache):
    def on_transcript(text):
        nonlocal avatar_image_path
        print("[User]:", text)
        reply = generate_reply(text, None)
        print("[Reply]:", reply)

        if HEYGEN_API_KEY and avatar_image_path:
            print("[HeyGen] Pipeline: upload -> create avatar -> generate video")
            tp_id = heygen_create_or_get_talking_photo(avatar_image_path, group_cache)
            if tp_id:
                vid = create_avatar_video_with_talking_photo(tp_id, reply)
                if vid:
                    print("[HeyGen] video_id:", vid)
                    total_wait = 0
                    while total_wait < 600:
                        st = check_video_status(vid)
                        if st and isinstance(st, dict):
                            data = st.get("data") or st
                            status = data.get("status") or data.get("state")
                            print(f"[HeyGen] video status: {status}")
                            if status and status.lower() == "completed":
                                video_url = get_video_url_from_status(st)
                                if video_url:
                                    print("[HeyGen] Video ready:", video_url)
                                    webbrowser.open(video_url)
                                    return
                                else:
                                    print("[HeyGen] video completed but no direct url found; response:", st)
                                    return
                            elif status and status.lower() in ("failed","error"):
                                print("[HeyGen] video generation failed:", st)
                                break
                        else:
                            print("[HeyGen] status polling returned empty; retrying")
                        time.sleep(4)
                        total_wait += 4
                    print("[HeyGen] timeout waiting for video generation. Check dashboard.")
                else:
                    print("[HeyGen] failed to start video generation.")
            else:
                print("[HeyGen] could not create or find talking photo for this image.")
        else:
            print("[Local TTS] HeyGen not configured or no avatar image set; playing local TTS.")
            tmpwav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_path = tmpwav.name; tmpwav.close()
            synthesize_tts_to_wav(reply, wav_path)
            wave_obj = sa.WaveObject.from_wave_file(wav_path)
            play_obj = wave_obj.play()
            while play_obj.is_playing():
                time.sleep(0.1)
            try:
                os.remove(wav_path)
            except:
                pass

    return on_transcript

# ---------------------------
# TTS helper
# ---------------------------
def synthesize_tts_to_wav(text, wav_path):
    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    engine.setProperty("rate", max(120, rate - 10))
    engine.save_to_file(text, wav_path)
    engine.runAndWait()

# ---------------------------
# Remaining code: UI loop (unchanged)
# ---------------------------
def generate_reply_with_groq(user_text, history=None, max_tokens=256):
    if not USE_GROQ:
        return None
    try:
        messages = [{"role": "user", "content": user_text}]
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content
        if isinstance(text, str):
            return text.strip()
        return str(text).strip()
    except Exception as e:
        print("[WARN] Groq call failed:", e)
        return None

def generate_reply(user_text, history=None):
    if USE_GROQ:
        out = generate_reply_with_groq(user_text, history)
        if out:
            return out
    u = user_text.lower()
    if any(x in u for x in ("hello", "hi", "hey")):
        return "Hi! I heard you. Ask me anything about the image."
    if "who" in u and "you" in u:
        return "I'm your friendly avatar created from your photo."
    if "age" in u:
        return "I can't determine exact ages, but you look great!"
    if "smile" in u or "happy" in u:
        return "Sure — I'll smile for you."
    return "Thanks for telling me. I'll do my best to answer."

class VoskListener:
    def __init__(self, model_path, samplerate=16000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at: {model_path}")
        self.model = Model(model_path)
        self.samplerate = samplerate
        self.rec = None
        self.q = queue.Queue()
        self.running = False
        self.thread = None
        self.result_callback = None

    def audio_callback(self, indata, frames, time_info, status):
        if self.rec is None:
            return
        try:
            data_bytes = indata.tobytes() if isinstance(indata, np.ndarray) else bytes(indata)
            if self.rec.AcceptWaveform(data_bytes):
                res = self.rec.Result()
                try:
                    j = json.loads(res)
                    text = j.get("text", "").strip()
                    if text:
                        self.q.put(text)
                except Exception:
                    pass
        except Exception as e:
            print("[VOSK] audio_callback error:", e)

    def start(self, result_callback):
        self.result_callback = result_callback
        self.rec = KaldiRecognizer(self.model, self.samplerate)
        self.running = True
        self.thread = threading.Thread(target=self._run_stream, daemon=True)
        self.thread.start()

    def _run_stream(self):
        try:
            with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000, dtype="int16",
                                   channels=1, callback=self.audio_callback):
                print("[VOSK] Listening... (speak now)")
                while self.running:
                    try:
                        text = self.q.get(timeout=0.5)
                        if self.result_callback:
                            try:
                                self.result_callback(text)
                            except Exception as e:
                                print("[VOSK] result_callback error:", e)
                    except queue.Empty:
                        continue
        except Exception as e:
            print("[VOSK] _run_stream error (input device?):", e)

    def stop(self):
        self.running = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", type=str, help="Optional initial avatar image path or URL")
    parser.add_argument("--test-key", action="store_true", help="Test HEYGEN_API_KEY and exit")
    args = parser.parse_args()

    print("=== Realtime Avatar Chatbot (HeyGen) ===")
    print("[INFO] HEYGEN_API_KEY (masked):", mask_key(HEYGEN_API_KEY))

    if args.test_key:
        ok, status, body = test_heygen_key_once(HEYGEN_API_KEY)
        print("HeyGen key test -> ok:", ok, "status:", status)
        print("Response body (truncated):")
        if isinstance(body, (dict, list)):
            print(json.dumps(body)[:2000])
        else:
            txt = str(body)
            print(txt[:2000])
        return

    if not os.path.exists(VOSK_MODEL_PATH):
        print("Vosk model missing at:", VOSK_MODEL_PATH)
        print("Download a small model (e.g. vosk-model-small-en-us-0.15) and set VOSK_MODEL_PATH in .env")
        return

    listener = VoskListener(VOSK_MODEL_PATH)
    avatar_bgr = np.zeros((480,640,3), dtype=np.uint8) + 120
    avatar_loaded = False
    avatar_path = None
    avatar_pts = None
    group_cache = {}

    if args.image:
        local = ensure_local_image(args.image)
        if local:
            img, pts = load_avatar_image(local)
            if img is not None:
                avatar_bgr = img; avatar_loaded = True; avatar_path = local; avatar_pts = pts
            else:
                print("[WARN] Could not load provided initial image. Continuing without avatar.")
        else:
            print("[WARN] Could not download or find initial image. Continuing without avatar.")

    on_transcript = on_transcript_factory(avatar_path, group_cache)

    def refresh_callback(new_path):
        nonlocal on_transcript, avatar_path
        avatar_path = new_path
        on_transcript = on_transcript_factory(avatar_path, group_cache)
        try:
            listener.result_callback = on_transcript
            print("[DEBUG] listener.result_callback updated to new on_transcript")
        except Exception as e:
            print("[WARN] could not update listener.result_callback:", e)
        return on_transcript

    listener.start(on_transcript)

    cv2.namedWindow("Avatar", cv2.WINDOW_AUTOSIZE)
    print("Controls: u=file picker  p=paste path or URL  q=quit")
    instr = np.zeros((240,480,3), dtype=np.uint8) + 40
    cv2.putText(instr, "Press 'u' (file) or 'p' (paste path/URL) to load image", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(instr, "Speak; HeyGen will generate a video for replies (if configured)", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    try:
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("u"):
                print("Opening file picker...")
                path = None
                if TK_AVAILABLE:
                    path = filedialog.askopenfilename(title="Select avatar image",
                                                      filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")])
                if not path:
                    print("File dialog canceled or not available.")
                else:
                    path = ensure_local_image(path)
                    img, pts = load_avatar_image(path)
                    if img is not None:
                        avatar_bgr = img; avatar_pts = pts; avatar_loaded = True; avatar_path = path
                        on_transcript = refresh_callback(avatar_path)
                        print("[UI] Avatar loaded:", path)
            elif key == ord("p"):
                print("Type image path or URL (absolute or relative) and press Enter (or just Enter to cancel):")
                path = input().strip().strip('"').strip("'")
                if not path:
                    print("Canceled.")
                else:
                    path = ensure_local_image(path)
                    if not path:
                        print("[ERROR] Could not fetch the given path/URL.")
                    else:
                        img, pts = load_avatar_image(path)
                        if img is not None:
                            avatar_bgr = img; avatar_pts = pts; avatar_loaded = True; avatar_path = path
                            on_transcript = refresh_callback(avatar_path)
                            print("[UI] Avatar loaded:", path)
            elif key == ord("q"):
                print("Quitting."); break

            display = avatar_bgr.copy() if avatar_loaded else instr
            cv2.imshow("Avatar", display)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        cv2.destroyAllWindows()
        try:
            if face_mesh: face_mesh.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
