import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$", re.IGNORECASE)
FALLBACK_NAME = "sin-titulo"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir(frozen: bool | None = None, env: Mapping[str, str] | None = None) -> Path:
    if frozen is None:
        frozen = is_frozen()
    if not frozen:
        return app_root() / "data"
    # The install dir under Program Files is read-only for normal users.
    env = os.environ if env is None else env
    base = env.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    return root / "YouTubeWhisperStudio"


def default_output_dir(frozen: bool | None = None, env: Mapping[str, str] | None = None) -> Path:
    if frozen is None:
        frozen = is_frozen()
    if not frozen:
        return app_root() / "outputs"
    env = os.environ if env is None else env
    profile = env.get("USERPROFILE")
    home = Path(profile) if profile else Path.home()
    return home / "Videos" / "YouTube Whisper Studio"


def bundled_bin_dir(frozen: bool | None = None) -> Path | None:
    """Directory with tools shipped inside the frozen bundle (deno for yt-dlp)."""
    if frozen is None:
        frozen = is_frozen()
    meipass = getattr(sys, "_MEIPASS", None)
    if not frozen or not meipass:
        return None
    candidate = Path(meipass) / "bin"
    return candidate if candidate.is_dir() else None


def sanitize_filename(name: str, max_len: int = 80) -> str:
    cleaned = _ILLEGAL.sub("", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned[:max_len].rstrip(". ")
    cleaned = cleaned.lstrip()
    if not cleaned or not cleaned.strip("."):
        return FALLBACK_NAME
    if _RESERVED.match(cleaned):
        cleaned = "_" + cleaned
    return cleaned


def job_output_dir(base: str | Path, title: str, ident: str) -> Path:
    return Path(base) / f"{sanitize_filename(title)}-[{ident}]"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
