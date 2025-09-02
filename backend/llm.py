# backend/llm.py
from groq import Groq
from dotenv import load_dotenv
import os
import requests
import logging

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_LLM_MODEL", "gemma2-9b-it").strip()

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in server environment (.env)")

# Priority list of fallback models to try if the configured model is unavailable.
# You can reorder or add other model ids from your account.
FALLBACK_PREFERENCES = [
    "gpt-4o",
    "compound-beta",
    "compound-beta-mini",
    "llama-3.1-8b-instant",
    "gemma2-9b-it"
]

client = Groq(api_key=GROQ_API_KEY)
logger = logging.getLogger("backend.llm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


def _fetch_available_models():
    """
    Fetch models from Groq 'models' endpoint. Returns a list of model ids (or empty list on error).
    """
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        # normalize
        models = []
        if isinstance(data, dict) and "data" in data:
            for m in data.get("data", []):
                if isinstance(m, dict) and m.get("id"):
                    models.append(m["id"])
        elif isinstance(data, list):
            for m in data:
                if isinstance(m, dict) and m.get("id"):
                    models.append(m["id"])
                elif isinstance(m, str):
                    models.append(m)
        elif isinstance(data, dict):
            models = list(data.keys())
        return models
    except Exception as e:
        logger.warning("Could not fetch available models: %s", e)
        return []


def _choose_model(available_models):
    """
    Decide which model to use:
      1) If MODEL_NAME set and available, return it.
      2) Else pick the first match from FALLBACK_PREFERENCES that exists in available_models.
      3) Else return None.
    """
    if MODEL_NAME:
        if MODEL_NAME in available_models:
            return MODEL_NAME
    # look for preferred fallbacks
    for pref in FALLBACK_PREFERENCES:
        if pref in available_models:
            return pref
    # if nothing matched, as a last resort, return the first available model if any
    if available_models:
        return available_models[0]
    return None


def get_llm_response(prompt: str) -> str:
    # try an optimistic call with configured model if present
    effective_model = MODEL_NAME or None

    # First attempt: if user configured model and we should try it directly
    if effective_model:
        try:
            logger.info("Attempting to use configured model: %s", effective_model)
            response = client.chat.completions.create(
                model=effective_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512
            )
            try:
                return response.choices[0].message.content
            except Exception:
                if isinstance(response, dict):
                    return response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return str(response)
        except Exception as e:
            err_text = str(e)
            logger.warning("Configured model '%s' failed: %s", effective_model, err_text)

    # If we reached here, configured model didn't work (or wasn't set). Try to fetch available models and pick a fallback.
    available = _fetch_available_models()
    if not available:
        raise RuntimeError(
            f"Configured model '{MODEL_NAME}' inaccessible and unable to fetch available models. "
            "Check GROQ_API_KEY, account permissions & network connectivity."
        )

    chosen = _choose_model(available)
    if not chosen:
        raise RuntimeError(
            f"Configured model '{MODEL_NAME}' inaccessible and no fallback model found. "
            f"Available models: {', '.join(available[:50])}"
        )

    # Try with chosen fallback model
    logger.info("Falling back to model: %s (available models sample: %s)", chosen, ", ".join(available[:10]))
    try:
        response = client.chat.completions.create(
            model=chosen,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        try:
            # If we got a response, return it and also inform via logs which model was used.
            ans = response.choices[0].message.content
            logger.info("Successfully used model: %s", chosen)
            return ans
        except Exception:
            if isinstance(response, dict):
                return response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return str(response)
    except Exception as e:
        # final failure: provide helpful diagnostics
        raise RuntimeError(
            f"LLM request failed using fallback model '{chosen}'. "
            f"Error: {e}\nAvailable models (sample): {', '.join(available[:30])}"
        ) from e
