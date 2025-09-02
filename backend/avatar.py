# backend/avatar.py
import sys
import subprocess
import os
from pathlib import Path
import logging

log = logging.getLogger("backend.avatar")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# Use WAV2LIP_DIR env var if set, else default absolute path
WAV2LIP_WORKDIR = Path(os.getenv("WAV2LIP_DIR", r"D:\auto_antimate\wav2lip"))

WAV2LIP_SCRIPT = WAV2LIP_WORKDIR / "inference.py"
_possible_ckpt_paths = [
    WAV2LIP_WORKDIR / "wav2lip_gan.pth",
    WAV2LIP_WORKDIR / "checkpoints" / "wav2lip_gan.pth",
    WAV2LIP_WORKDIR / "checkpoints" / "Wav2Lip_gan.pth",
    WAV2LIP_WORKDIR / "models" / "wav2lip_gan.pth"
]

def _find_checkpoint():
    for p in _possible_ckpt_paths:
        if p.exists():
            log.info("Using Wav2Lip checkpoint: %s", p)
            return p
    log.warning("Wav2Lip checkpoint not found. Searched:\n  %s", "\n  ".join(str(p) for p in _possible_ckpt_paths))
    return None

WAV2LIP_CKPT = _find_checkpoint()

def generate_talking_video(face_path: str, audio_path: str, out_video_path: str, extra_args=None, timeout=300):
    """
    Runs Wav2Lip inference as subprocess and returns out_video_path.
    Raises informative exceptions on failure.
    """
    if not WAV2LIP_SCRIPT.exists():
        raise FileNotFoundError(f"Wav2Lip inference script not found at {WAV2LIP_SCRIPT}. "
                                "Ensure you cloned Wav2Lip into WAV2LIP_WORKDIR or set WAV2LIP_DIR env var.")

    if WAV2LIP_CKPT is None:
        raise FileNotFoundError(
            "Wav2Lip checkpoint not found. Please place 'wav2lip_gan.pth' in one of:\n  " +
            "\n  ".join(str(p) for p in _possible_ckpt_paths)
        )

    cmd = [
        sys.executable, str(WAV2LIP_SCRIPT),
        "--checkpoint_path", str(WAV2LIP_CKPT),
        "--face", str(face_path),
        "--audio", str(audio_path),
        "--outfile", str(out_video_path)
    ]
    if extra_args:
        cmd += list(extra_args)

    log.info("Running Wav2Lip: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, cwd=str(WAV2LIP_WORKDIR), check=True, capture_output=True, text=True, timeout=timeout)
        log.info("Wav2Lip stdout:\n%s", res.stdout)
        log.info("Wav2Lip stderr:\n%s", res.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Wav2Lip failed (code={e.returncode}). stdout:\n{e.stdout}\nstderr:\n{e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Wav2Lip timed out after {timeout}s. Partial stdout/stderr:\n{e.stdout}\n{e.stderr}") from e

    if not Path(out_video_path).exists():
        raise RuntimeError("Avatar generation failed: output missing after Wav2Lip run")

    return out_video_path
