import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from whisper_studio.models import PlaylistSlot

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$", re.IGNORECASE)
FALLBACK_NAME = "sin-titulo"
DATA_DIR_NAME = "Subtitula"
LEGACY_DATA_DIR_NAME = "YouTubeWhisperStudio"


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
    return root / DATA_DIR_NAME


def migrate_legacy_data_dir(frozen: bool | None = None, env: Mapping[str, str] | None = None) -> str:
    """Moves the pre-rebrand data dir to its new name; returns "migrated", "not_needed" or "failed"."""
    if frozen is None:
        frozen = is_frozen()
    if not frozen:
        return "not_needed"
    target = data_dir(frozen=True, env=env)
    legacy = target.with_name(LEGACY_DATA_DIR_NAME)
    if target.exists() or not legacy.is_dir():
        return "not_needed"
    try:
        # A rename keeps the multi-GB models in place; on failure the old dir is left untouched.
        legacy.rename(target)
    except OSError:
        return "failed"
    return "migrated"


def default_output_dir(frozen: bool | None = None, env: Mapping[str, str] | None = None) -> Path:
    if frozen is None:
        frozen = is_frozen()
    if not frozen:
        return app_root() / "outputs"
    env = os.environ if env is None else env
    profile = env.get("USERPROFILE")
    home = Path(profile) if profile else Path.home()
    return home / "Videos" / "Subtitula"


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


# Windows tools (yt-dlp, ffmpeg) fail with "No such file or directory" past MAX_PATH (260), so the
# job folder leaves room for the longest file written inside it: ".tmp\" plus a 60-byte media name
# with its id, format and ".part" suffixes.
MAX_OUTPUT_DIR_LEN = 150
_GROUP_NAME_LEN = 60
_MIN_TITLE_LEN = 20


def job_output_dir(
    base: str | Path,
    title: str,
    ident: str,
    playlist: PlaylistSlot | None = None,
    folder: tuple[str, ...] = (),
) -> Path:
    # Group folders are capped independently of the title so every job of a playlist or added
    # folder lands in the same parent; only the per-job title is trimmed to fit the budget.
    if playlist is None:
        groups = [sanitize_filename(part, _GROUP_NAME_LEN) for part in folder]
        prefix = ""
    else:
        groups = [f"{sanitize_filename(playlist.title, _GROUP_NAME_LEN)}-[{playlist.id}]"]
        prefix = f"{str(playlist.index).zfill(max(2, len(str(playlist.total))))} - "
    parent = Path(base).joinpath(*groups)
    suffix = f"-[{ident}]"
    room = MAX_OUTPUT_DIR_LEN - len(str(parent)) - 1 - len(prefix) - len(suffix)
    name = sanitize_filename(title, max(_MIN_TITLE_LEN, min(80, room)))
    return parent / f"{prefix}{name}{suffix}"


def legacy_job_output_dir(
    base: str | Path,
    title: str,
    ident: str,
    playlist: PlaylistSlot | None = None,
    folder: tuple[str, ...] = (),
) -> Path:
    """Folder used before the path budget, so jobs finished back then are still recognised."""
    if playlist is None:
        parent = Path(base).joinpath(*(sanitize_filename(part) for part in folder))
        return parent / f"{sanitize_filename(title)}-[{ident}]"
    number = str(playlist.index).zfill(max(2, len(str(playlist.total))))
    group = Path(base) / f"{sanitize_filename(playlist.title)}-[{playlist.id}]"
    return group / f"{number} - {sanitize_filename(title)}-[{ident}]"


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
