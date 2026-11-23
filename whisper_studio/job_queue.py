import copy
import json
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from whisper_studio import media
from whisper_studio.config import AppConfig
from whisper_studio.downloader import DownloadProgress, DownloadResult
from whisper_studio.errors import CancelledError, DownloadFailed, EngineError
from whisper_studio.exporters import ExportMeta, export_all, to_json, to_txt
from whisper_studio.logging_setup import get_logger
from whisper_studio.models import (
    ACTIVE_STATUSES,
    Job,
    JobKind,
    JobStatus,
    Segment,
    TranscriptResult,
)
from whisper_studio.paths import atomic_write_text, job_output_dir
from whisper_studio.transcriber import EngineInfo

log = get_logger("queue")

TEMP_DIR_NAME = ".tmp"
METADATA_NAME = "metadata.json"
_PROGRESS_EMIT_INTERVAL_S = 0.1


class EngineProtocol(Protocol):
    def ensure_loaded(
        self, model_name: str, device_pref: str, on_status: Callable[[str], None]
    ) -> EngineInfo: ...

    def transcribe(
        self,
        source: Any,
        language: str | None,
        word_timestamps: bool,
        vad_filter: bool,
        on_segment: Callable[[Segment, float | None], None],
        cancel_event: threading.Event,
    ) -> TranscriptResult: ...


class DownloaderProtocol(Protocol):
    def download(
        self,
        url: str,
        mode: str,
        quality: str,
        out_dir: Path,
        on_progress: Callable[[DownloadProgress], None],
        cancel_event: threading.Event,
    ) -> DownloadResult: ...


@dataclass(frozen=True)
class QueueEvent:
    """Base class; the UI dispatches on the concrete subclass."""


@dataclass(frozen=True)
class JobAdded(QueueEvent):
    job: Job


@dataclass(frozen=True)
class JobUpdated(QueueEvent):
    job: Job


@dataclass(frozen=True)
class SegmentEvent(QueueEvent):
    job_id: str
    segment: Segment


@dataclass(frozen=True)
class EngineStatus(QueueEvent):
    info: EngineInfo | None
    text: str


@dataclass(frozen=True)
class JobFinished(QueueEvent):
    job: Job


@dataclass(frozen=True)
class QueueFinished(QueueEvent):
    pass


@dataclass(frozen=True)
class Notice(QueueEvent):
    level: str  # "info" | "warning" | "error"
    message: str


def _local_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


class JobQueue:
    def __init__(
        self,
        engine: EngineProtocol,
        downloader: DownloaderProtocol,
        duration_probe: Callable[[Path], float | None] = media.probe_duration,
        ffmpeg_available: bool | None = None,
    ) -> None:
        self._engine = engine
        self._downloader = downloader
        self._probe_duration = duration_probe
        self._ffmpeg_override = ffmpeg_available
        self.events: queue.Queue[QueueEvent] = queue.Queue()
        self._jobs: list[Job] = []
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._current: Job | None = None
        self._current_cancel: threading.Event | None = None
        self._config = AppConfig()
        self._notified_fallbacks: set[str] = set()
        self._created_out_dir = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._worker is not None

    def jobs(self) -> list[Job]:
        with self._lock:
            return [copy.copy(j) for j in self._jobs]

    def _emit(self, event: QueueEvent) -> None:
        self.events.put(event)

    def _notice(self, level: str, message: str) -> None:
        self._emit(Notice(level, message))

    def add_youtube(
        self,
        url: str,
        video_id: str,
        title: str,
        mode: str,
        duration: float | None = None,
        thumbnail_url: str | None = None,
    ) -> Job | None:
        with self._lock:
            if any(
                j.video_id == video_id and self._counts_for_dedup(j) for j in self._jobs
            ):
                self._notice("warning", f"«{title}» ya está en la lista.")
                return None
            job = Job(
                kind=JobKind.YOUTUBE, source=url, title=title, mode=mode, video_id=video_id,
                duration=duration, thumbnail_url=thumbnail_url, stage_text="En cola",
            )
            self._jobs.append(job)
            snapshot = copy.copy(job)
        self._emit(JobAdded(snapshot))
        return snapshot

    def add_local(self, path: Path) -> Job | None:
        path = Path(path)
        key = _local_key(path)
        with self._lock:
            if any(
                j.kind == JobKind.LOCAL and _local_key(Path(j.source)) == key and self._counts_for_dedup(j)
                for j in self._jobs
            ):
                self._notice("warning", f"«{path.name}» ya está en la lista.")
                return None
            job = Job(
                kind=JobKind.LOCAL, source=str(path), title=path.stem,
                mode="audio_transcribe", stage_text="En cola",
            )
            self._jobs.append(job)
            snapshot = copy.copy(job)
        self._emit(JobAdded(snapshot))
        return snapshot

    @staticmethod
    def _counts_for_dedup(job: Job) -> bool:
        return job.status in ACTIVE_STATUSES or job.status in (
            JobStatus.PENDING, JobStatus.COMPLETED, JobStatus.SKIPPED,
        )

    def remove(self, job_id: str) -> bool:
        with self._lock:
            for job in self._jobs:
                if job.id == job_id:
                    if job is self._current or job.status in ACTIVE_STATUSES:
                        return False
                    self._jobs.remove(job)
                    return True
        return False

    def clear_finished(self) -> int:
        with self._lock:
            keep = [j for j in self._jobs if not j.is_finished]
            removed = len(self._jobs) - len(keep)
            self._jobs = keep
            return removed

    def start(self, config: AppConfig) -> bool:
        """Start the worker, or hand a fresh config to the running one. Returns True if a thread was started."""
        with self._lock:
            self._config = copy.deepcopy(config)
            if self._worker is not None:
                return False
            self._worker = threading.Thread(target=self._run_worker, name="job-queue", daemon=True)
            self._worker.start()
            return True

    def cancel_current(self) -> None:
        with self._lock:
            if self._current_cancel is not None:
                self._current_cancel.set()

    def cancel_all(self) -> None:
        cancelled: list[Job] = []
        with self._lock:
            if self._current_cancel is not None:
                self._current_cancel.set()
            for job in self._jobs:
                if job is not self._current and job.status == JobStatus.PENDING:
                    job.status = JobStatus.CANCELLED
                    job.stage_text = "Cancelado"
                    cancelled.append(copy.copy(job))
        for snapshot in cancelled:
            self._emit(JobUpdated(snapshot))
            self._emit(JobFinished(snapshot))

    def shutdown(self, timeout: float = 5.0) -> None:
        self.cancel_all()
        with self._lock:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def _next_pending(self) -> Job | None:
        for job in self._jobs:
            if job.status == JobStatus.PENDING:
                return job
        return None

    def _run_worker(self) -> None:
        while True:
            with self._lock:
                job = self._next_pending()
                if job is None:
                    self._current = None
                    self._current_cancel = None
                    self._emit(QueueFinished())
                    self._worker = None
                    return
                self._current = job
                self._current_cancel = threading.Event()
                cancel = self._current_cancel
                config = self._config
            self._process(job, config, cancel)

    def _publish(self, job: Job) -> None:
        self._emit(JobUpdated(copy.copy(job)))

    def _set(self, job: Job, status: JobStatus, stage: str, **fields: Any) -> None:
        job.status = status
        job.stage_text = stage
        for name, value in fields.items():
            setattr(job, name, value)
        self._publish(job)

    def _process(self, job: Job, config: AppConfig, cancel: threading.Event) -> None:
        job.started_at = time.monotonic()
        self._created_out_dir = False
        try:
            self._run_job(job, config, cancel)
        except CancelledError:
            self._set(job, JobStatus.CANCELLED, "Cancelado", speed_text="")
        except (EngineError, DownloadFailed) as exc:
            log.error("Job %s failed: %s (%s)", job.title, exc.user_message, exc.detail)
            self._set(job, JobStatus.ERROR, "Error", error=exc.user_message, speed_text="")
        except Exception as exc:
            log.exception("Unexpected failure in job %s", job.title)
            self._set(job, JobStatus.ERROR, "Error", error=f"Error inesperado: {exc}", speed_text="")
        finally:
            job.finished_at = time.monotonic()
            self._cleanup(job)
            with self._lock:
                self._current = None
                self._current_cancel = None
            self._emit(JobFinished(copy.copy(job)))

    def _cleanup(self, job: Job) -> None:
        """Remove only this job's temp download folder and an output folder it created if still empty."""
        out_dir = job.output_dir
        if out_dir is None:
            return
        shutil.rmtree(out_dir / TEMP_DIR_NAME, ignore_errors=True)
        if job.status in (JobStatus.CANCELLED, JobStatus.ERROR) and self._created_out_dir:
            try:
                out_dir.rmdir()
            except OSError:
                pass

    def _is_complete(self, out_dir: Path) -> bool:
        try:
            data = json.loads((out_dir / METADATA_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(data, dict) and data.get("status") == "completed"

    def _run_job(self, job: Job, config: AppConfig, cancel: threading.Event) -> None:
        needs_transcription = job.mode != "download_only" or job.kind == JobKind.LOCAL
        if job.kind == JobKind.YOUTUBE:
            ident = job.video_id or "video"
        else:
            local = Path(job.source)
            if not local.is_file():
                raise EngineError("No se encontró el archivo de audio o video.", detail=job.source)
            ident = media.file_fingerprint(local)
        out_dir = job_output_dir(config.output_dir, job.title, ident)
        job.output_dir = out_dir
        self._created_out_dir = not out_dir.exists()

        if needs_transcription and config.existing_policy == "skip" and self._is_complete(out_dir):
            self._set(job, JobStatus.SKIPPED, "Ya existe una transcripción", progress=1.0)
            return

        if job.kind == JobKind.YOUTUBE:
            media_path = self._download(job, config, cancel, out_dir)
            if job.mode == "download_only":
                self._set(job, JobStatus.COMPLETED, "Descarga completada", progress=1.0, speed_text="")
                return
        else:
            media_path = Path(job.source)

        self._check_cancel(cancel)
        self._set(job, JobStatus.PREPARING, "Preparando", progress=None, speed_text="")
        if job.duration is None:
            job.duration = self._probe_duration(media_path)
        if job.duration is None:
            self._set(job, JobStatus.PREPARING, "Preparando (duración desconocida: progreso indeterminado)")
        self._load_engine(config)
        self._check_cancel(cancel)

        result = self._transcribe(job, config, cancel, media_path)
        self._export(job, config, result, out_dir)
        self._set(job, JobStatus.COMPLETED, "Completado", progress=1.0, progress_is_estimate=False)

    @staticmethod
    def _check_cancel(cancel: threading.Event) -> None:
        if cancel.is_set():
            raise CancelledError()

    def _download(
        self, job: Job, config: AppConfig, cancel: threading.Event, out_dir: Path
    ) -> Path:
        # Choosing "video + transcription" is itself the request to keep the video.
        use_temp = job.mode == "audio_transcribe" and not config.keep_media
        target = out_dir / TEMP_DIR_NAME if use_temp else out_dir
        self._set(job, JobStatus.DOWNLOADING, "Descargando", progress=None, progress_is_estimate=False)
        last_emit = 0.0

        def on_progress(p: DownloadProgress) -> None:
            nonlocal last_emit
            now = time.monotonic()
            if p.fraction != 1.0 and now - last_emit < _PROGRESS_EMIT_INTERVAL_S:
                return
            last_emit = now
            job.progress = p.fraction
            job.progress_is_estimate = p.is_estimate
            job.speed_text = f"{p.speed_text} · ETA {p.eta_text}".strip(" ·") if p.eta_text else p.speed_text
            self._publish(job)

        result = self._downloader.download(
            job.source, job.mode, config.video_quality, target, on_progress, cancel
        )
        return result.path

    def _load_engine(self, config: AppConfig) -> EngineInfo:
        def on_status(text: str) -> None:
            self._emit(EngineStatus(None, text))

        info = self._engine.ensure_loaded(config.model, config.device, on_status)
        self._emit(
            EngineStatus(info, f"Modelo {info.model} en {info.device.upper()} ({info.compute_type})")
        )
        if info.fallback_reason and info.fallback_reason not in self._notified_fallbacks:
            self._notified_fallbacks.add(info.fallback_reason)
            self._notice("warning", info.fallback_reason)
        return info

    def _transcribe(
        self, job: Job, config: AppConfig, cancel: threading.Event, media_path: Path
    ) -> TranscriptResult:
        self._set(job, JobStatus.TRANSCRIBING, "Transcribiendo", progress=None, progress_is_estimate=False)
        last_emit = 0.0

        def on_segment(segment: Segment, progress: float | None) -> None:
            nonlocal last_emit
            job.progress = progress
            self._emit(SegmentEvent(job.id, segment))
            now = time.monotonic()
            if now - last_emit >= _PROGRESS_EMIT_INTERVAL_S:
                last_emit = now
                self._publish(job)

        return self._engine.transcribe(
            media_path, config.language, config.word_timestamps, config.vad_filter, on_segment, cancel
        )

    def _export(self, job: Job, config: AppConfig, result: TranscriptResult, out_dir: Path) -> None:
        meta = ExportMeta(
            title=job.title,
            source_url=job.source if job.kind == JobKind.YOUTUBE else None,
            duration=job.duration or result.duration or None,
            language=result.language,
            model=result.model,
            device=result.device,
            created_at=datetime.now(),
        )
        text_formats = [f for f in config.formats if f != "json"]
        export_all(result, meta, out_dir, text_formats)
        extra = {
            "status": "completed",
            "source": job.source,
            "video_id": job.video_id,
            "mode": job.mode,
            "formats": list(config.formats),
        }
        # Written last so a crash can never leave a "completed" marker next to missing files.
        atomic_write_text(
            out_dir / METADATA_NAME,
            to_json(result, meta, include_segments="json" in config.formats, extra=extra),
        )
        job.transcript_text = to_txt(result)
