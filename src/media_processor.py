"""
src/media_processor.py
=======================
Audio and video → text transcription using faster-whisper (offline, CPU).
ffmpeg is used to extract audio from video files via subprocess.

Supported:
  Audio : .mp3, .wav, .m4a, .ogg, .flac, .aac, .opus
  Video : .mp4, .avi, .mov, .mkv, .webm, .3gp

Returns: (transcript_text, detected_language_code)
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp"}

WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium | large-v3


def _load_whisper():
    """Lazy-load faster-whisper model (downloads once, cached locally)."""
    try:
        from faster_whisper import WhisperModel
        print(f"[MediaProcessor] Loading faster-whisper '{WHISPER_MODEL_SIZE}'...")
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("[MediaProcessor] ✓ Whisper model ready")
        return model
    except ImportError:
        raise ImportError(
            "faster-whisper not installed. Run: pip install faster-whisper"
        )


def _check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def _extract_audio_from_video(video_path: str) -> str:
    """
    Use ffmpeg subprocess to extract audio stream from video.
    Returns path to a temporary WAV file.
    """
    if not _check_ffmpeg():
        raise EnvironmentError(
            "ffmpeg not found on PATH.\n"
            "Install it:\n"
            "  Windows : https://ffmpeg.org/download.html → add bin/ to PATH\n"
            "  Linux   : sudo apt install ffmpeg\n"
            "  macOS   : brew install ffmpeg"
        )

    tmp_audio = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",  # 16-bit PCM WAV
        "-ar", "16000",           # 16kHz sample rate (Whisper requirement)
        "-ac", "1",               # mono
        tmp_audio
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed:\n{err}")

    return tmp_audio


def _transcribe_audio(audio_path: str, whisper_model) -> tuple[str, str]:
    """
    Transcribe an audio file using faster-whisper.
    Returns (transcript_text, language_code).
    """
    segments, info = whisper_model.transcribe(
        audio_path,
        beam_size=5,
        language=None,       # auto-detect language
        vad_filter=True,     # voice activity detection to skip silence
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(seg.text.strip() for seg in segments)
    lang = info.language
    prob = info.language_probability
    print(f"[MediaProcessor] Language detected: '{lang}' ({prob:.2f} confidence)")
    return text.strip(), lang


# ── Public API ────────────────────────────────────────────────────────────────

_whisper_model = None   # lazy singleton


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = _load_whisper()
    return _whisper_model


def process_audio(audio_path: str) -> tuple[str, str]:
    """
    Transcribe an audio file to text.

    Args:
        audio_path: Path to audio file (.mp3 / .wav / .m4a / .ogg / .flac / etc.)

    Returns:
        (transcript_text, detected_language_code)

    Raises:
        FileNotFoundError  if file does not exist
        ValueError         if file extension is unsupported
        ImportError        if faster-whisper is not installed
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = path.suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    print(f"[MediaProcessor] Transcribing audio: {path.name}")
    text, lang = _transcribe_audio(str(path), _get_whisper())

    if not text:
        raise ValueError(
            "Transcription produced no output. "
            "Check that the audio contains speech."
        )

    print(f"[MediaProcessor] ✓ Transcript ({len(text)} chars): {text[:100]}...")
    return text, lang


def process_video(video_path: str) -> tuple[str, str]:
    """
    Extract audio from a video file and transcribe to text.

    Args:
        video_path: Path to video file (.mp4 / .avi / .mov / .mkv / .webm)

    Returns:
        (transcript_text, detected_language_code)

    Raises:
        FileNotFoundError  if file does not exist
        ValueError         if file extension is unsupported
        EnvironmentError   if ffmpeg is not on PATH
        RuntimeError       if ffmpeg extraction fails
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ext = path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{ext}'. "
            f"Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    print(f"[MediaProcessor] Extracting audio from video: {path.name}")
    tmp_audio = None
    try:
        tmp_audio = _extract_audio_from_video(str(path))
        print(f"[MediaProcessor] ✓ Audio extracted → {tmp_audio}")
        text, lang = _transcribe_audio(tmp_audio, _get_whisper())
    finally:
        # Always clean up temp file
        if tmp_audio and os.path.exists(tmp_audio):
            os.remove(tmp_audio)

    if not text:
        raise ValueError(
            "No speech detected in video. "
            "Ensure the video contains audible speech."
        )

    print(f"[MediaProcessor] ✓ Transcript ({len(text)} chars): {text[:100]}...")
    return text, lang


def process_media(file_path: str) -> tuple[str, str]:
    """
    Auto-detect file type (audio or video) and transcribe.

    Args:
        file_path: Path to any supported audio or video file.

    Returns:
        (transcript_text, detected_language_code)
    """
    ext = Path(file_path).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return process_audio(file_path)
    elif ext in VIDEO_EXTENSIONS:
        return process_video(file_path)
    else:
        raise ValueError(
            f"Unknown file type '{ext}'. "
            f"Audio: {', '.join(sorted(AUDIO_EXTENSIONS))} | "
            f"Video: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )
