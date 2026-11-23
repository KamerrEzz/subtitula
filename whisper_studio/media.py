import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm"})
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"})
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

_PROBE_TIMEOUT_S = 30


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS


def discover_media(folder: Path, recursive: bool = True) -> list[Path]:
    if not folder.is_dir():
        return []
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in iterator if p.is_file() and is_media_file(p))


def which_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def which_ffprobe() -> str | None:
    return shutil.which("ffprobe")


def probe_duration(path: Path) -> float | None:
    """Duration in seconds via ffprobe, or None when unavailable or unreadable."""
    exe = which_ffprobe()
    if exe is None:
        return None
    cmd = [
        exe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
            creationflags=flags, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        value = float(proc.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def check_free_space(path: Path, required_bytes: int) -> bool:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free >= required_bytes
    except OSError:
        return True


def file_fingerprint(path: Path) -> str:
    resolved = path.resolve()
    st = resolved.stat()
    key = f"{resolved}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
