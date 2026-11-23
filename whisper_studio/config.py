import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from whisper_studio.logging_setup import get_logger
from whisper_studio.paths import atomic_write_text, data_dir, default_output_dir

log = get_logger("config")

MODELS = ("tiny", "base", "small", "medium", "large-v3")
LANGUAGES = ("es", "en", "auto")
DEVICES = ("auto", "cuda", "cpu")
FORMATS = ("txt", "srt", "vtt", "md", "json")
DOWNLOAD_MODES = ("audio_transcribe", "video_transcribe", "download_only")
VIDEO_QUALITIES = ("best", "1080", "720", "480", "360")
EXISTING_POLICIES = ("skip", "regenerate")

_CHOICES: dict[str, tuple[str, ...]] = {
    "model": MODELS,
    "language": LANGUAGES,
    "device": DEVICES,
    "download_mode": DOWNLOAD_MODES,
    "video_quality": VIDEO_QUALITIES,
    "existing_policy": EXISTING_POLICIES,
}


def default_model_for(cuda_ok: bool) -> str:
    return "medium" if cuda_ok else "small"


def default_settings_path() -> Path:
    return data_dir() / "settings.json"


@dataclass
class AppConfig:
    output_dir: str = field(default_factory=lambda: str(default_output_dir()))
    model: str = "small"
    model_user_set: bool = False
    language: str = "es"
    device: str = "auto"
    formats: list[str] = field(default_factory=lambda: list(FORMATS))
    download_mode: str = "audio_transcribe"
    video_quality: str = "best"
    keep_media: bool = False
    word_timestamps: bool = False
    vad_filter: bool = True
    readable_subtitles: bool = True
    existing_policy: str = "skip"
    window_geometry: str | None = None


def _coerce(name: str, value: Any, default: Any) -> Any:
    if name in _CHOICES:
        return value if isinstance(value, str) and value in _CHOICES[name] else default
    if name == "formats":
        if not isinstance(value, list):
            return default
        chosen = [f for f in FORMATS if f in value]
        return chosen or default
    if name in ("keep_media", "word_timestamps", "vad_filter", "model_user_set", "readable_subtitles"):
        return value if isinstance(value, bool) else default
    if name == "output_dir":
        return value if isinstance(value, str) and value.strip() else default
    if name == "window_geometry":
        return value if isinstance(value, str) or value is None else default
    return default


def _back_up_corrupt(path: Path) -> None:
    backup = path.with_name(path.name + ".bak")
    try:
        os.replace(path, backup)
        log.warning("Corrupt settings file moved to %s", backup)
    except OSError:
        log.warning("Corrupt settings file %s could not be backed up", path, exc_info=True)


def load_config(path: Path | None = None) -> AppConfig:
    path = path or default_settings_path()
    defaults = AppConfig()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return defaults
    except OSError:
        log.warning("Settings file %s unreadable; using defaults", path, exc_info=True)
        return defaults
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        _back_up_corrupt(path)
        return defaults
    values: dict[str, Any] = {}
    for f in fields(AppConfig):
        if f.name in data:
            default = getattr(defaults, f.name)
            values[f.name] = _coerce(f.name, data[f.name], default)
            if values[f.name] is default and data[f.name] != default:
                log.warning("Invalid value for setting %r replaced by default", f.name)
    return AppConfig(**{**asdict(defaults), **values})


def save_config(cfg: AppConfig, path: Path | None = None) -> None:
    path = path or default_settings_path()
    atomic_write_text(path, json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
