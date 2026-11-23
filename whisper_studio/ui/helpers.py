"""Pure, Tk-free helpers for the UI so they can be unit tested."""

import re
from dataclasses import dataclass
from pathlib import Path

from whisper_studio.models import ACTIVE_STATUSES, FINISHED_STATUSES, Job, JobStatus
from whisper_studio.transcriber import EngineInfo
from whisper_studio.ui import theme

_STATUS_COLORS: dict[JobStatus, str] = {
    JobStatus.PENDING: theme.MUTED,
    JobStatus.DOWNLOADING: theme.BLUE,
    JobStatus.PREPARING: theme.WARNING,
    JobStatus.TRANSCRIBING: theme.ACCENT,
    JobStatus.COMPLETED: theme.SUCCESS,
    JobStatus.ERROR: theme.ERROR,
    JobStatus.CANCELLED: theme.WARNING,
    JobStatus.SKIPPED: theme.BLUE,
}

_DROP_TOKEN = re.compile(r"\{([^}]*)\}|(\S+)")
_URL_TOKEN = re.compile(r"\S+")
_BARE_YOUTUBE = re.compile(r"^(www\.|m\.)?(youtube\.com|youtu\.be|youtube-nocookie\.com)/", re.IGNORECASE)


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"[{hours}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def ellipsize(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def status_color(status: JobStatus) -> str:
    return _STATUS_COLORS[status]


def parse_drop_paths(data: str) -> list[Path]:
    """Parse tkinterdnd2's Tcl list ('{C:/a b.mp4} C:/c.wav') into paths."""
    paths: list[Path] = []
    for match in _DROP_TOKEN.finditer(data):
        raw = match.group(1) if match.group(1) is not None else match.group(2)
        if raw.strip():
            paths.append(Path(raw))
    return paths


def parse_urls(text: str) -> list[str]:
    """Extract unique http(s) links from pasted text; bare youtube hosts get https:// prepended."""
    urls: list[str] = []
    for token in _URL_TOKEN.findall(text):
        if _BARE_YOUTUBE.match(token):
            token = "https://" + token
        if token.lower().startswith(("http://", "https://")) and token not in urls:
            urls.append(token)
    return urls


@dataclass(frozen=True)
class QueueSummary:
    text: str
    fraction: float
    pending: int
    active: int


def summarize_jobs(jobs: list[Job]) -> QueueSummary:
    total = len(jobs)
    if total == 0:
        return QueueSummary("Sin trabajos", 0.0, 0, 0)
    done = sum(1 for j in jobs if j.status in (JobStatus.COMPLETED, JobStatus.SKIPPED))
    errors = sum(1 for j in jobs if j.status == JobStatus.ERROR)
    pending = sum(1 for j in jobs if j.status == JobStatus.PENDING)
    active = sum(1 for j in jobs if j.status in ACTIVE_STATUSES)
    finished = sum(1 for j in jobs if j.status in FINISHED_STATUSES)
    partial = sum(j.progress or 0.0 for j in jobs if j.status in ACTIVE_STATUSES)
    text = f"{done} de {total} completados"
    if errors:
        text += f" · {errors} error" + ("es" if errors > 1 else "")
    return QueueSummary(text, min(1.0, (finished + partial) / total), pending, active)


@dataclass
class DeviceState:
    checked: bool = False
    cuda_detected: bool = False
    gpu_name: str = ""


def device_badge(state: DeviceState, info: EngineInfo | None) -> tuple[str, str]:
    if info is not None:
        if info.device == "cuda":
            name = f"GPU · {state.gpu_name} · " if state.gpu_name else "GPU · "
            return f"{name}{info.compute_type}", theme.SUCCESS
        if info.fallback_reason:
            return f"CPU · {info.compute_type} (CUDA falló)", theme.WARNING
        return f"CPU · {info.compute_type}", theme.BLUE
    if not state.checked:
        return "Detectando…", theme.MUTED
    if state.cuda_detected:
        return "CUDA detectado (se verificará al cargar el modelo)", theme.BLUE
    return "Sin GPU CUDA · se usará la CPU", theme.MUTED


def engine_badge(info: EngineInfo | None, loading_model: str | None, failed: bool) -> tuple[str, str]:
    if failed:
        return "Error", theme.ERROR
    if loading_model:
        return f"Cargando modelo {loading_model}…", theme.WARNING
    if info is not None:
        return f"Listo · {info.model}", theme.SUCCESS
    return "Motor sin cargar", theme.MUTED
