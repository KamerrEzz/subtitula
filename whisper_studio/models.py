import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class JobStatus(StrEnum):
    PENDING = "Pendiente"
    DOWNLOADING = "Descargando"
    PREPARING = "Preparando"
    TRANSCRIBING = "Transcribiendo"
    COMPLETED = "Completado"
    ERROR = "Error"
    CANCELLED = "Cancelado"
    SKIPPED = "Omitido"


FINISHED_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.CANCELLED, JobStatus.SKIPPED}
)
ACTIVE_STATUSES = frozenset({JobStatus.DOWNLOADING, JobStatus.PREPARING, JobStatus.TRANSCRIBING})


class JobKind(StrEnum):
    YOUTUBE = "youtube"
    LOCAL = "local"


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] | None = None


@dataclass
class TranscriptResult:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float
    model: str
    device: str
    compute_type: str


@dataclass(frozen=True)
class PlaylistSlot:
    id: str
    title: str
    index: int
    total: int


@dataclass
class Job:
    kind: JobKind
    source: str
    title: str
    mode: str
    video_id: str | None = None
    duration: float | None = None
    thumbnail_url: str | None = None
    playlist: PlaylistSlot | None = None
    folder: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.PENDING
    stage_text: str = ""
    progress: float | None = None
    progress_is_estimate: bool = False
    speed_text: str = ""
    error: str = ""
    output_dir: Path | None = None
    transcript_text: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    @property
    def is_finished(self) -> bool:
        return self.status in FINISHED_STATUSES

