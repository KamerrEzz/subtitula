import json
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from whisper_studio import exporters, job_queue
from whisper_studio.config import AppConfig
from whisper_studio.downloader import DownloadProgress, DownloadResult
from whisper_studio.errors import CancelledError, DownloadFailed, EngineError
from whisper_studio.job_queue import (
    EngineStatus,
    JobAdded,
    JobFinished,
    JobQueue,
    JobUpdated,
    Notice,
    QueueFinished,
    SegmentEvent,
)
from whisper_studio.models import Job, JobStatus, PlaylistSlot, Segment, TranscriptResult, Word
from whisper_studio.paths import job_output_dir
from whisper_studio.transcriber import EngineInfo


class FakeEngine:
    def __init__(self, info: EngineInfo | None = None) -> None:
        self.info = info or EngineInfo("small", "cuda", "int8_float16")
        self.sources: list[str] = []
        self.word_flags: list[bool] = []
        self.load_calls = 0
        self.behaviors: dict[str, Exception] = {}
        self.on_start: Callable[[str], None] | None = None

    def ensure_loaded(self, model: str, device: str, on_status: Callable[[str], None]) -> EngineInfo:
        self.load_calls += 1
        on_status("cargando")
        return self.info

    def transcribe(self, source, language, word_timestamps, vad_filter, on_segment, cancel_event):
        name = Path(str(source)).stem
        self.sources.append(name)
        if self.on_start:
            self.on_start(name)
        if name in self.behaviors:
            raise self.behaviors[name]
        self.word_flags.append(word_timestamps)
        segments = [Segment(0.0, 1.0, "Hola."), Segment(1.0, 2.0, "Mundo.")]
        if word_timestamps:
            segments = [Segment(s.start, s.end, s.text, [Word(s.start, s.end, " " + s.text, 0.9)]) for s in segments]
        for seg in segments:
            on_segment(seg, seg.end / 2.0)
            if cancel_event.is_set():
                raise CancelledError()
        return TranscriptResult(
            segments, "es", 0.99, 2.0, self.info.model, self.info.device, self.info.compute_type
        )


class FakeDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Path]] = []
        self.errors: dict[str, Exception] = {}

    def download(self, url, mode, quality, out_dir, on_progress, cancel_event) -> DownloadResult:
        self.calls.append((url, mode, Path(out_dir)))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vid = url.rsplit("=", 1)[-1]
        (out_dir / f"{vid}.m4a.part").write_bytes(b"p")
        if vid in self.errors:
            raise self.errors[vid]
        on_progress(DownloadProgress(0.5, False, "1.0 MB/s", "0:05", 5, 10))
        (out_dir / f"{vid}.m4a.part").unlink()
        media = out_dir / f"{vid}.m4a"
        media.write_bytes(b"audio")
        on_progress(DownloadProgress(1.0, False, "", "", 10, 10))
        return DownloadResult(media, {"id": vid})


@pytest.fixture
def env(tmp_path: Path):
    engine = FakeEngine()
    downloader = FakeDownloader()
    out = tmp_path / "out"
    cfg = AppConfig(output_dir=str(out))
    q = JobQueue(engine, downloader, duration_probe=lambda p: 10.0, ffmpeg_available=True)
    return q, engine, downloader, cfg, tmp_path


def make_media(tmp_path: Path, name: str) -> Path:
    p = tmp_path / "src" / f"{name}.wav"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"RIFF" + name.encode())
    return p


def run_to_end(q: JobQueue, cfg: AppConfig, timeout: float = 10.0) -> list[Any]:
    q.start(cfg)
    events: list[Any] = []
    while True:
        try:
            ev = q.events.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError("queue did not finish") from None
        events.append(ev)
        if isinstance(ev, QueueFinished):
            return events


def statuses(q: JobQueue) -> list[JobStatus]:
    return [j.status for j in q.jobs()]


def test_sequential_processing_and_outputs(env) -> None:
    q, engine, _, cfg, tmp = env
    for n in ("a", "b", "c"):
        q.add_local(make_media(tmp, n))
    events = run_to_end(q, cfg)
    assert engine.sources == ["a", "b", "c"]
    assert statuses(q) == [JobStatus.COMPLETED] * 3
    kinds = {type(e) for e in events}
    assert {JobAdded, JobUpdated, SegmentEvent, EngineStatus, JobFinished, QueueFinished} <= kinds
    job = q.jobs()[0]
    assert job.duration == 10.0 and job.progress == 1.0 and job.transcript_text
    names = sorted(p.name for p in job.output_dir.iterdir())
    assert names == ["metadata.json", "transcripcion.md", "transcripcion.srt", "transcripcion.txt", "transcripcion.vtt"]
    assert (tmp / "src" / "a.wav").read_bytes() == b"RIFFa"


def test_metadata_written_last_and_content(env, monkeypatch: pytest.MonkeyPatch) -> None:
    q, _, _, cfg, tmp = env
    written: list[str] = []
    real = exporters.atomic_write_text

    def spy(path: Path, text: str) -> None:
        written.append(path.name)
        real(path, text)

    monkeypatch.setattr(exporters, "atomic_write_text", spy)
    monkeypatch.setattr(job_queue, "atomic_write_text", spy)
    q.add_local(make_media(tmp, "a"))
    run_to_end(q, cfg)
    assert written[-1] == "metadata.json" and written.count("metadata.json") == 1
    meta = json.loads((q.jobs()[0].output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed"
    assert meta["model"] == "small" and meta["device"] == "cuda" and meta["compute_type"] == "int8_float16"
    assert meta["formats"] == ["txt", "srt", "vtt", "md", "json"]
    assert meta["language"] == "es" and meta["title"] == "a"
    assert len(meta["segments"]) == 2


def test_metadata_without_segments_when_json_not_selected(env) -> None:
    q, _, _, cfg, tmp = env
    cfg.formats = ["txt"]
    q.add_local(make_media(tmp, "a"))
    run_to_end(q, cfg)
    out = q.jobs()[0].output_dir
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert "segments" not in meta and meta["status"] == "completed"
    assert sorted(p.name for p in out.iterdir()) == ["metadata.json", "transcripcion.txt"]


def test_duplicate_local_rejected(env) -> None:
    q, _, _, _, tmp = env
    p = make_media(tmp, "a")
    assert q.add_local(p) is not None
    assert q.add_local(p) is None
    notices = []
    while not q.events.empty():
        ev = q.events.get_nowait()
        if isinstance(ev, Notice):
            notices.append(ev)
    assert notices and notices[0].level == "warning"
    assert len(q.jobs()) == 1


def test_duplicate_youtube_by_video_id(env) -> None:
    q, *_ = env
    assert q.add_youtube("https://youtu.be/abc", "abc", "T", "audio_transcribe") is not None
    assert q.add_youtube("https://www.youtube.com/watch?v=abc", "abc", "T", "audio_transcribe") is None


def test_error_in_second_job_continues(env) -> None:
    q, engine, _, cfg, tmp = env
    for n in ("a", "b", "c"):
        q.add_local(make_media(tmp, n))
    engine.behaviors["b"] = EngineError("La GPU se quedó sin memoria.", "detail")
    run_to_end(q, cfg)
    jobs = q.jobs()
    assert [j.status for j in jobs] == [JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.COMPLETED]
    assert jobs[1].error == "La GPU se quedó sin memoria."
    assert engine.sources == ["a", "b", "c"]


def test_unexpected_error_message(env) -> None:
    q, engine, _, cfg, tmp = env
    q.add_local(make_media(tmp, "a"))
    engine.behaviors["a"] = RuntimeError("boom")
    run_to_end(q, cfg)
    job = q.jobs()[0]
    assert job.status == JobStatus.ERROR and job.error == "Error inesperado: boom"


def test_missing_local_file_is_error(env) -> None:
    q, _, _, cfg, tmp = env
    p = make_media(tmp, "a")
    q.add_local(p)
    p.unlink()
    run_to_end(q, cfg)
    job = q.jobs()[0]
    assert job.status == JobStatus.ERROR and "archivo" in job.error


def test_cancel_current_mid_transcription_then_next_runs(env) -> None:
    q, engine, _, cfg, tmp = env
    first = make_media(tmp, "a")
    for n in ("a", "b"):
        q.add_local(make_media(tmp, n))
    engine.on_start = lambda name: q.cancel_current() if name == "a" else None
    run_to_end(q, cfg)
    jobs = q.jobs()
    assert [j.status for j in jobs] == [JobStatus.CANCELLED, JobStatus.COMPLETED]
    assert not jobs[0].output_dir.exists()
    assert first.exists()


def test_cancel_all_stops_queue(env) -> None:
    q, engine, _, cfg, tmp = env
    for n in ("a", "b", "c"):
        q.add_local(make_media(tmp, n))
    engine.on_start = lambda name: q.cancel_all()
    run_to_end(q, cfg)
    assert statuses(q) == [JobStatus.CANCELLED] * 3
    assert engine.sources == ["a"]


def test_skip_existing_completed(env) -> None:
    q, engine, _, cfg, tmp = env
    p = make_media(tmp, "a")
    from whisper_studio.media import file_fingerprint

    out = job_output_dir(cfg.output_dir, "a", file_fingerprint(p))
    out.mkdir(parents=True)
    (out / "metadata.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    q.add_local(p)
    run_to_end(q, cfg)
    job = q.jobs()[0]
    assert job.status == JobStatus.SKIPPED
    assert job.stage_text == "Ya existe una transcripción"
    assert engine.sources == []


def test_regenerate_policy_reprocesses(env) -> None:
    q, engine, _, cfg, tmp = env
    cfg.existing_policy = "regenerate"
    p = make_media(tmp, "a")
    from whisper_studio.media import file_fingerprint

    out = job_output_dir(cfg.output_dir, "a", file_fingerprint(p))
    out.mkdir(parents=True)
    (out / "metadata.json").write_text(json.dumps({"status": "completed", "old": True}), encoding="utf-8")
    q.add_local(p)
    run_to_end(q, cfg)
    assert q.jobs()[0].status == JobStatus.COMPLETED
    assert "old" not in json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def test_incomplete_metadata_is_not_skipped(env) -> None:
    q, engine, _, cfg, tmp = env
    p = make_media(tmp, "a")
    from whisper_studio.media import file_fingerprint

    out = job_output_dir(cfg.output_dir, "a", file_fingerprint(p))
    out.mkdir(parents=True)
    (out / "metadata.json").write_text("{broken", encoding="utf-8")
    q.add_local(p)
    run_to_end(q, cfg)
    assert q.jobs()[0].status == JobStatus.COMPLETED


def test_youtube_audio_temp_deleted_when_not_kept(env) -> None:
    q, engine, dl, cfg, _ = env
    job = q.add_youtube("https://www.youtube.com/watch?v=abc", "abc", "Mi video", "audio_transcribe", duration=30)
    run_to_end(q, cfg)
    job = q.jobs()[0]
    out = job_output_dir(cfg.output_dir, "Mi video", "abc")
    assert job.status == JobStatus.COMPLETED and job.output_dir == out
    assert not (out / ".tmp").exists()
    assert not list(out.glob("*.m4a"))
    assert (out / "transcripcion.txt").exists()
    assert dl.calls[0][2] == out / ".tmp"
    assert engine.sources == ["abc"]


def test_playlist_videos_are_written_inside_the_playlist_folder(env) -> None:
    q, _, _, cfg, _ = env
    for index, vid in enumerate(("abc", "def"), start=1):
        slot = PlaylistSlot(id="PL1", title="Curso", index=index, total=2)
        q.add_youtube(f"https://www.youtube.com/watch?v={vid}", vid, f"Clase {index}", "audio_transcribe", playlist=slot)
    run_to_end(q, cfg)
    folder = Path(cfg.output_dir) / "Curso-[PL1]"
    assert sorted(p.name for p in folder.iterdir()) == ["01 - Clase 1-[abc]", "02 - Clase 2-[def]"]
    assert all((d / "transcripcion.txt").exists() for d in folder.iterdir())


def test_youtube_audio_kept(env) -> None:
    q, _, dl, cfg, _ = env
    cfg.keep_media = True
    q.add_youtube("https://www.youtube.com/watch?v=abc", "abc", "Mi video", "audio_transcribe")
    run_to_end(q, cfg)
    out = job_output_dir(cfg.output_dir, "Mi video", "abc")
    assert (out / "abc.m4a").exists()
    assert dl.calls[0][2] == out


def test_video_mode_keeps_video_even_without_keep_media(env) -> None:
    q, _, dl, cfg, _ = env
    cfg.keep_media = False
    q.add_youtube("https://www.youtube.com/watch?v=abc", "abc", "Mi video", "video_transcribe")
    run_to_end(q, cfg)
    out = job_output_dir(cfg.output_dir, "Mi video", "abc")
    assert dl.calls[0][2] == out
    assert q.jobs()[0].status == JobStatus.COMPLETED


def test_download_only_skips_engine(env) -> None:
    q, engine, _, cfg, _ = env
    q.add_youtube("https://www.youtube.com/watch?v=abc", "abc", "V", "download_only")
    run_to_end(q, cfg)
    job = q.jobs()[0]
    out = job_output_dir(cfg.output_dir, "V", "abc")
    assert job.status == JobStatus.COMPLETED
    assert (out / "abc.m4a").exists()
    assert engine.load_calls == 0 and engine.sources == []


def test_download_failure_cleans_temp_and_continues(env) -> None:
    q, engine, dl, cfg, tmp = env
    dl.errors["bad"] = DownloadFailed("El video es privado y no se puede descargar.", "d")
    q.add_youtube("https://www.youtube.com/watch?v=bad", "bad", "Malo", "audio_transcribe")
    q.add_local(make_media(tmp, "a"))
    run_to_end(q, cfg)
    jobs = q.jobs()
    assert jobs[0].status == JobStatus.ERROR
    assert jobs[0].error == "El video es privado y no se puede descargar."
    assert not job_output_dir(cfg.output_dir, "Malo", "bad").exists()
    assert jobs[1].status == JobStatus.COMPLETED


def test_local_files_from_an_added_folder_are_grouped_under_it(env) -> None:
    q, _, _, cfg, tmp = env
    a, b = make_media(tmp, "M0S1-a"), make_media(tmp, "M0S2-b")
    q.add_local(a, folder=("git",))
    q.add_local(b, folder=("git",))
    run_to_end(q, cfg)
    group = Path(cfg.output_dir) / "git"
    assert sorted(p.name.split("-[")[0] for p in group.iterdir()) == ["M0S1-a", "M0S2-b"]
    assert q.jobs()[0].output_dir.parent == group


def test_failed_local_file_leaves_no_empty_group_folders(env) -> None:
    q, engine, _, cfg, tmp = env
    engine.behaviors["bad"] = EngineError("Falló.", detail="x")
    q.add_local(make_media(tmp, "bad"), folder=("git", "modulo1"))
    run_to_end(q, cfg)
    assert q.jobs()[0].status == JobStatus.ERROR
    assert not (Path(cfg.output_dir) / "git").exists()


def test_failed_playlist_video_leaves_no_empty_playlist_folder(env) -> None:
    q, _, dl, cfg, _ = env
    dl.errors["bad"] = DownloadFailed("El video es privado y no se puede descargar.", "d")
    slot = PlaylistSlot(id="PL1", title="Curso", index=1, total=1)
    q.add_youtube("https://www.youtube.com/watch?v=bad", "bad", "Malo", "audio_transcribe", playlist=slot)
    run_to_end(q, cfg)
    assert q.jobs()[0].status == JobStatus.ERROR
    assert not (Path(cfg.output_dir) / "Curso-[PL1]").exists()


def test_jobs_finished_with_the_previous_folder_names_are_still_skipped(env) -> None:
    q, engine, _, cfg, _ = env
    cfg.existing_policy = "skip"
    long_title = "M10S3 - MakeFile con IA comandos para todo el equipo y un título muy largo"
    slot = PlaylistSlot(id="PL1", title="Curso de OpenCode — Agentes de IA para Programar", index=1, total=1)
    old = Path(cfg.output_dir) / "Curso de OpenCode — Agentes de IA para Programar-[PL1]" / f"01 - {long_title}-[abc]"
    old.mkdir(parents=True)
    (old / "metadata.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    q.add_youtube("https://www.youtube.com/watch?v=abc", "abc", long_title, "audio_transcribe", playlist=slot)
    run_to_end(q, cfg)
    job = q.jobs()[0]
    assert job.status == JobStatus.SKIPPED and job.output_dir == old
    assert engine.sources == []


def test_retry_failed_requeues_only_failed_jobs_and_they_can_complete(env) -> None:
    q, engine, _, cfg, tmp = env
    engine.behaviors["a"] = EngineError("Falló.", detail="x")
    q.add_local(make_media(tmp, "a"))
    q.add_local(make_media(tmp, "b"))
    run_to_end(q, cfg)
    assert [j.status for j in q.jobs()] == [JobStatus.ERROR, JobStatus.COMPLETED]

    del engine.behaviors["a"]
    assert q.retry_failed() == 1
    retried = q.jobs()[0]
    assert retried.status == JobStatus.PENDING and retried.error == "" and retried.progress is None
    run_to_end(q, cfg)
    assert [j.status for j in q.jobs()] == [JobStatus.COMPLETED, JobStatus.COMPLETED]


def test_retry_failed_without_failures_does_nothing(env) -> None:
    q, _, _, _, _ = env
    assert q.retry_failed() == 0


def test_download_progress_is_forwarded(env) -> None:
    q, _, _, cfg, _ = env
    q.add_youtube("https://www.youtube.com/watch?v=abc", "abc", "V", "audio_transcribe")
    events = run_to_end(q, cfg)
    downloading = [
        e.job for e in events if isinstance(e, JobUpdated) and e.job.status == JobStatus.DOWNLOADING
    ]
    assert any(j.progress == 0.5 and j.speed_text == "1.0 MB/s · ETA 0:05" for j in downloading)


def test_fallback_notice_once(env) -> None:
    q, engine, _, cfg, tmp = env
    engine.info = EngineInfo("small", "cpu", "int8", fallback_reason="Se usará la CPU.", cuda_error_detail="x")
    for n in ("a", "b"):
        q.add_local(make_media(tmp, n))
    events = run_to_end(q, cfg)
    notices = [e for e in events if isinstance(e, Notice) and "CPU" in e.message]
    assert len(notices) == 1
    assert any(isinstance(e, EngineStatus) and e.info and e.info.device == "cpu" for e in events)


def test_unknown_duration_keeps_progress_indeterminate(tmp_path: Path) -> None:
    engine = FakeEngine()
    q = JobQueue(engine, FakeDownloader(), duration_probe=lambda p: None, ffmpeg_available=False)
    cfg = AppConfig(output_dir=str(tmp_path / "out"))
    q.add_local(make_media(tmp_path, "a"))
    events = run_to_end(q, cfg)
    assert q.jobs()[0].duration is None
    preparing = [e.job for e in events if isinstance(e, JobUpdated) and e.job.status == JobStatus.PREPARING]
    assert preparing and all(j.progress is None for j in preparing)


def test_remove_and_clear_finished(env) -> None:
    q, _, _, cfg, tmp = env
    a = q.add_local(make_media(tmp, "a"))
    b = q.add_local(make_media(tmp, "b"))
    assert q.remove(a.id) is True
    assert q.remove("missing") is False
    assert [j.id for j in q.jobs()] == [b.id]
    run_to_end(q, cfg)
    assert q.clear_finished() == 1
    assert q.jobs() == []


def test_remove_running_job_refused(env) -> None:
    q, engine, _, cfg, tmp = env
    a = q.add_local(make_media(tmp, "a"))
    results: list[bool] = []
    engine.on_start = lambda name: results.append(q.remove(a.id))
    run_to_end(q, cfg)
    assert results == [False]
    assert len(q.jobs()) == 1


def test_jobs_added_while_running_are_processed(env) -> None:
    q, engine, _, cfg, tmp = env
    q.add_local(make_media(tmp, "a"))

    def add_more(name: str) -> None:
        if name == "a":
            q.add_local(make_media(tmp, "late"))

    engine.on_start = add_more
    run_to_end(q, cfg)
    assert engine.sources == ["a", "late"]


def test_shutdown_cancels_and_joins(env) -> None:
    q, engine, _, cfg, tmp = env
    gate = threading.Event()
    release = threading.Event()

    def block(name: str) -> None:
        gate.set()
        release.wait(5)

    engine.on_start = block
    q.add_local(make_media(tmp, "a"))
    q.add_local(make_media(tmp, "b"))
    q.start(cfg)
    assert gate.wait(5)
    threading.Timer(0.2, release.set).start()
    q.shutdown(timeout=5)
    assert not q.is_running
    assert statuses(q) == [JobStatus.CANCELLED, JobStatus.CANCELLED]
    assert engine.sources == ["a"]


def test_job_snapshots_are_copies(env) -> None:
    q, *_, tmp = env
    q.add_local(make_media(tmp, "a"))
    snap = q.jobs()[0]
    snap.status = JobStatus.ERROR
    assert q.jobs()[0].status == JobStatus.PENDING
    assert isinstance(snap, Job)


def test_srt_selected_with_readable_subtitles_requests_words_even_if_switch_is_off(env) -> None:
    q, engine, _, cfg, tmp_path = env
    cfg.word_timestamps = False
    cfg.readable_subtitles = True
    cfg.formats = ["txt", "srt", "json"]
    q.add_local(make_media(tmp_path, "a"))
    run_to_end(q, cfg)
    assert engine.word_flags == [True]
    meta = json.loads((q.jobs()[0].output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert all("words" not in seg for seg in meta["segments"])


def test_words_stay_in_json_when_the_switch_is_on(env) -> None:
    q, engine, _, cfg, tmp_path = env
    cfg.word_timestamps = True
    cfg.formats = ["srt", "json"]
    q.add_local(make_media(tmp_path, "a"))
    run_to_end(q, cfg)
    meta = json.loads((q.jobs()[0].output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert all("words" in seg for seg in meta["segments"])


@pytest.mark.parametrize(
    "readable,formats",
    [(False, ["srt"]), (True, ["txt", "md", "json"])],
)
def test_words_not_requested_when_not_needed(env, readable: bool, formats: list[str]) -> None:
    q, engine, _, cfg, tmp_path = env
    cfg.word_timestamps = False
    cfg.readable_subtitles = readable
    cfg.formats = formats
    q.add_local(make_media(tmp_path, "a"))
    run_to_end(q, cfg)
    assert engine.word_flags == [False]
