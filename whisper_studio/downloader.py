import re
import socket
import threading
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from whisper_studio.errors import CancelledError, DownloadFailed
from whisper_studio.logging_setup import get_logger
from whisper_studio.media import check_free_space, which_ffmpeg

log = get_logger("downloader")

_YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
_UNAVAILABLE_TITLES = ("[private video]", "[deleted video]", "[unavailable video]")
_PARTIAL_NAME = re.compile(r"(\.part(-Frag\d+)?|\.ytdl|\.temp\.\w+)$|\.f\d[\w-]*\.\w+$")


@dataclass(frozen=True)
class ProbeEntry:
    id: str
    title: str
    url: str
    duration: float | None


@dataclass
class ProbeResult:
    kind: str  # "video" | "playlist"
    id: str
    title: str
    duration: float | None = None
    thumbnail: str | None = None
    webpage_url: str = ""
    entries: list[ProbeEntry] = field(default_factory=list)
    unavailable_count: int = 0
    playlist_hint: bool = False


@dataclass(frozen=True)
class DownloadProgress:
    fraction: float | None
    is_estimate: bool
    speed_text: str
    eta_text: str
    downloaded_bytes: int | None
    total_bytes: int | None


@dataclass
class DownloadResult:
    path: Path
    info: dict[str, Any]


ProgressCallback = Callable[[DownloadProgress], None]


def _ydl_class() -> Any:
    from yt_dlp import YoutubeDL

    return YoutubeDL


def is_supported_url(text: str) -> bool:
    try:
        parsed = urlparse(text.strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    return any(host == h or host.endswith("." + h) for h in _YOUTUBE_HOSTS)


def _is_pure_playlist_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    query = parse_qs(parsed.query)
    return "list" in query and "v" not in query and not parsed.hostname == "youtu.be"


def _has_list_param(url: str) -> bool:
    return "list" in parse_qs(urlparse(url.strip()).query)


class _YdlLogger:
    def debug(self, msg: str) -> None:
        if not msg.startswith("[debug]"):
            log.debug(msg)

    def info(self, msg: str) -> None:
        log.debug(msg)

    def warning(self, msg: str) -> None:
        log.warning(msg)

    def error(self, msg: str) -> None:
        log.error(msg)


def map_ytdlp_error(exc: BaseException) -> DownloadFailed:
    detail = f"{type(exc).__name__}: {exc}"
    lowered = str(exc).lower()
    inner = getattr(exc, "exc_info", None)
    inner_exc = inner[1] if inner and len(inner) > 1 else None
    network_types = (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError)
    if "private video" in lowered:
        message = "El video es privado y no se puede descargar."
    elif any(t in lowered for t in ("sign in", "log in", "login required", "confirm your age", "age-restricted", "age restricted")):
        message = "El video requiere iniciar sesión; no se omiten restricciones de acceso."
    elif any(t in lowered for t in ("your country", "geo restricted", "geo-restricted", "geographic")):
        message = "El video no está disponible en tu país."
    elif any(t in lowered for t in ("has been removed", "no longer available", "been terminated", "does not exist", "deleted")):
        message = "El video fue eliminado y ya no está disponible."
    elif "unavailable" in lowered or "not available" in lowered:
        message = "El video no está disponible."
    elif (
        isinstance(exc, network_types)
        or isinstance(inner_exc, network_types)
        or any(t in lowered for t in ("unable to download webpage", "getaddrinfo", "timed out", "connection", "network is unreachable"))
    ):
        message = "No hay conexión a internet o el servidor no responde. Inténtalo de nuevo."
    else:
        message = f"No se pudo procesar el video: {str(exc).strip() or type(exc).__name__}"
    return DownloadFailed(message, detail)


def _entry_to_result(entry: Any) -> ProbeEntry | None:
    if not isinstance(entry, dict) or not entry.get("id"):
        return None
    title = str(entry.get("title") or "")
    if title.lower() in _UNAVAILABLE_TITLES or entry.get("availability") in ("private", "needs_auth"):
        return None
    url = entry.get("url") or entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry['id']}"
    duration = entry.get("duration")
    return ProbeEntry(str(entry["id"]), title or str(entry["id"]), str(url), float(duration) if duration else None)


def probe(url: str) -> ProbeResult:
    pure_playlist = _is_pure_playlist_url(url)
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": not pure_playlist,
        "extract_flat": "in_playlist",
        "socket_timeout": 30,
        "logger": _YdlLogger(),
    }
    try:
        with _ydl_class()(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        log.warning("Probe failed for %s", url, exc_info=True)
        raise map_ytdlp_error(exc) from exc
    if not isinstance(info, dict):
        raise DownloadFailed("No se pudo obtener información del enlace.", "extract_info returned no data")
    hint = _has_list_param(url) and not pure_playlist
    if info.get("_type") == "playlist" or "entries" in info:
        entries: list[ProbeEntry] = []
        unavailable = 0
        for raw in info.get("entries") or []:
            parsed = _entry_to_result(raw)
            if parsed is None:
                unavailable += 1
            else:
                entries.append(parsed)
        return ProbeResult(
            kind="playlist",
            id=str(info.get("id") or ""),
            title=str(info.get("title") or "Lista de reproducción"),
            webpage_url=str(info.get("webpage_url") or url),
            entries=entries,
            unavailable_count=unavailable,
        )
    duration = info.get("duration")
    return ProbeResult(
        kind="video",
        id=str(info.get("id") or ""),
        title=str(info.get("title") or info.get("id") or "Video"),
        duration=float(duration) if duration else None,
        thumbnail=info.get("thumbnail"),
        webpage_url=str(info.get("webpage_url") or url),
        playlist_hint=hint,
    )


def _format_selector(mode: str, quality: str, ffmpeg_available: bool) -> str:
    if mode == "audio_transcribe":
        return "bestaudio[ext=m4a]/bestaudio/best"
    cap = "" if quality == "best" else f"[height<={quality}]"
    if ffmpeg_available:
        if quality == "best":
            return "bestvideo+bestaudio/best"
        return f"bestvideo{cap}+bestaudio/best{cap}"
    if quality == "best":
        return "best[acodec!=none][vcodec!=none]/best"
    return f"best{cap}[acodec!=none][vcodec!=none]/best"


def build_ydl_options(
    mode: str,
    quality: str,
    out_dir: Path,
    ffmpeg_available: bool,
    progress_hook: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "format": _format_selector(mode, quality, ffmpeg_available),
        "outtmpl": str(Path(out_dir) / "%(title).60B [%(id)s].%(ext)s"),
        "windowsfilenames": True,
        "restrictfilenames": False,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,
        "overwrites": False,
        "writethumbnail": False,
        "socket_timeout": 30,
        "logger": _YdlLogger(),
        "progress_hooks": [progress_hook],
    }
    if mode != "audio_transcribe" and ffmpeg_available:
        options["merge_output_format"] = "mp4"
    return options


def format_speed(value: float | None) -> str:
    if not value or value <= 0:
        return ""
    if value < 1024:
        return f"{value:.0f} B/s"
    if value < 1024**2:
        return f"{value / 1024:.1f} KB/s"
    return f"{value / 1024**2:.1f} MB/s"


def _format_eta(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return ""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def make_progress_hook(
    on_progress: ProgressCallback, cancel_event: threading.Event
) -> Callable[[dict[str, Any]], None]:
    def hook(d: dict[str, Any]) -> None:
        if cancel_event.is_set():
            raise CancelledError()
        status = d.get("status")
        downloaded = d.get("downloaded_bytes")
        if status == "finished":
            total = d.get("total_bytes") or downloaded
            on_progress(DownloadProgress(1.0, False, "", "", downloaded, total))
            return
        if status != "downloading":
            return
        total = d.get("total_bytes")
        estimate = False
        if not total:
            total = d.get("total_bytes_estimate")
            estimate = bool(total)
        fraction = None
        if total and downloaded is not None:
            fraction = min(1.0, max(0.0, downloaded / total))
        on_progress(
            DownloadProgress(
                fraction, estimate, format_speed(d.get("speed")), _format_eta(d.get("eta")),
                downloaded, int(total) if total else None,
            )
        )

    return hook


def _known_size(info: dict[str, Any]) -> int:
    streams = info.get("requested_formats") or [info]
    total = 0
    for stream in streams:
        total += int(stream.get("filesize") or stream.get("filesize_approx") or 0)
    return total


def _snapshot(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()} if directory.is_dir() else set()


def _remove_new_partials(directory: Path, before: set[str]) -> None:
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.name not in before and path.is_file() and _PARTIAL_NAME.search(path.name):
            try:
                path.unlink()
            except OSError:
                log.warning("Could not remove partial file %s", path, exc_info=True)


def download(
    url: str,
    mode: str,
    quality: str,
    out_dir: Path,
    on_progress: ProgressCallback,
    cancel_event: threading.Event,
    ffmpeg_available: bool | None = None,
) -> DownloadResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cancel_event.is_set():
        raise CancelledError()
    if ffmpeg_available is None:
        ffmpeg_available = which_ffmpeg() is not None
    before = _snapshot(out_dir)
    hook = make_progress_hook(on_progress, cancel_event)
    options = build_ydl_options(mode, quality, out_dir, ffmpeg_available, hook)
    try:
        with _ydl_class()(options) as ydl:
            info = ydl.extract_info(url, download=False)
            size = _known_size(info)
            if size and not check_free_space(out_dir, size):
                raise DownloadFailed(
                    "Espacio insuficiente en disco.", f"required about {size} bytes"
                )
            result = ydl.process_ie_result(info, download=True)
            path = _final_path(ydl, result)
    except CancelledError:
        _remove_new_partials(out_dir, before)
        raise
    except DownloadFailed:
        raise
    except Exception as exc:
        if cancel_event.is_set():
            _remove_new_partials(out_dir, before)
            raise CancelledError() from exc
        _remove_new_partials(out_dir, before)
        log.warning("Download failed for %s", url, exc_info=True)
        raise map_ytdlp_error(exc) from exc
    if not path.exists():
        raise DownloadFailed(
            "La descarga terminó pero no se encontró el archivo.", f"expected {path}"
        )
    return DownloadResult(path, result)


def _final_path(ydl: Any, result: dict[str, Any]) -> Path:
    downloads = result.get("requested_downloads") or []
    if downloads and downloads[0].get("filepath"):
        return Path(downloads[0]["filepath"])
    return Path(ydl.prepare_filename(result))


class YtDlpDownloader:
    """Default downloader implementation injected into JobQueue."""

    def download(
        self,
        url: str,
        mode: str,
        quality: str,
        out_dir: Path,
        on_progress: ProgressCallback,
        cancel_event: threading.Event,
    ) -> DownloadResult:
        return download(url, mode, quality, out_dir, on_progress, cancel_event)
