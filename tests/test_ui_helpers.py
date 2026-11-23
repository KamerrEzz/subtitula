from pathlib import Path

import pytest

from whisper_studio.models import Job, JobKind, JobStatus
from whisper_studio.transcriber import EngineInfo
from whisper_studio.ui import theme
from whisper_studio.ui.helpers import (
    DeviceState,
    device_badge,
    ellipsize,
    engine_badge,
    format_clock,
    format_elapsed,
    parse_drop_paths,
    parse_urls,
    status_color,
    summarize_jobs,
)


def make_job(status: JobStatus, progress: float | None = None) -> Job:
    job = Job(kind=JobKind.LOCAL, source="x", title="t", mode="audio_transcribe")
    job.status = status
    job.progress = progress
    return job


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (5.9, "0:05"), (65, "1:05"), (3599, "59:59"), (3600, "1:00:00"), (3725, "1:02:05"), (-3, "0:00")],
)
def test_format_elapsed(seconds: float, expected: str) -> None:
    assert format_elapsed(seconds) == expected


def test_format_clock_pads_minutes_and_adds_hours() -> None:
    assert format_clock(7) == "[00:07]"
    assert format_clock(754) == "[12:34]"
    assert format_clock(3725) == "[1:02:05]"


def test_ellipsize() -> None:
    assert ellipsize("short", 10) == "short"
    assert ellipsize("abcdefghij", 10) == "abcdefghij"
    assert ellipsize("abcdefghijk", 10) == "abcdefghi…"


def test_status_color_covers_every_status() -> None:
    for status in JobStatus:
        assert status_color(status).startswith("#")
    assert status_color(JobStatus.ERROR) == theme.ERROR
    assert status_color(JobStatus.COMPLETED) == theme.SUCCESS
    assert status_color(JobStatus.TRANSCRIBING) == theme.ACCENT
    assert status_color(JobStatus.PENDING) == theme.MUTED


def test_parse_drop_paths_braced_and_plain() -> None:
    data = r"{C:/Users/me/My Videos/a b.mp4} C:/x/c.wav {D:/folder with space}"
    assert parse_drop_paths(data) == [
        Path("C:/Users/me/My Videos/a b.mp4"),
        Path("C:/x/c.wav"),
        Path("D:/folder with space"),
    ]


def test_parse_drop_paths_empty() -> None:
    assert parse_drop_paths("") == []
    assert parse_drop_paths("   ") == []


def test_parse_urls_multiline_dedup_and_scheme_repair() -> None:
    text = (
        "https://www.youtube.com/watch?v=aaaaaaaaaaa\n"
        "  https://youtu.be/bbbbbbbbbbb  \r\n"
        "youtu.be/ccccccccccc\n"
        "https://www.youtube.com/watch?v=aaaaaaaaaaa\n"
        "no es un enlace\n"
    )
    assert parse_urls(text) == [
        "https://www.youtube.com/watch?v=aaaaaaaaaaa",
        "https://youtu.be/bbbbbbbbbbb",
        "https://youtu.be/ccccccccccc",
    ]


def test_parse_urls_keeps_other_http_links_for_validation() -> None:
    assert parse_urls("https://example.com/x") == ["https://example.com/x"]


def test_parse_urls_empty() -> None:
    assert parse_urls("") == []


def test_summarize_jobs_counts_and_fraction() -> None:
    jobs = [
        make_job(JobStatus.COMPLETED),
        make_job(JobStatus.SKIPPED),
        make_job(JobStatus.ERROR),
        make_job(JobStatus.TRANSCRIBING, 0.5),
        make_job(JobStatus.PENDING),
    ]
    summary = summarize_jobs(jobs)
    assert summary.text == "2 de 5 completados · 1 error"
    assert summary.fraction == pytest.approx((3 + 0.5) / 5)
    assert summary.pending == 1
    assert summary.active == 1


def test_summarize_jobs_plural_errors_and_cancelled() -> None:
    jobs = [make_job(JobStatus.ERROR), make_job(JobStatus.ERROR), make_job(JobStatus.CANCELLED)]
    summary = summarize_jobs(jobs)
    assert summary.text == "0 de 3 completados · 2 errores"
    assert summary.pending == 0


def test_summarize_jobs_empty() -> None:
    summary = summarize_jobs([])
    assert summary.text == "Sin trabajos"
    assert summary.fraction == 0.0


def test_device_badge_states() -> None:
    assert device_badge(DeviceState(), None)[0] == "Detectando…"
    cuda = DeviceState(checked=True, cuda_detected=True, gpu_name="NVIDIA GeForce GTX 1650")
    assert device_badge(cuda, None)[0] == "CUDA detectado (se verificará al cargar el modelo)"
    verified = device_badge(cuda, EngineInfo("medium", "cuda", "int8_float16"))
    assert verified == ("GPU · NVIDIA GeForce GTX 1650 · int8_float16", theme.SUCCESS)
    cpu_only = DeviceState(checked=True, cuda_detected=False)
    assert device_badge(cpu_only, None)[0] == "Sin GPU CUDA · se usará la CPU"
    assert device_badge(cpu_only, EngineInfo("small", "cpu", "int8")) == ("CPU · int8", theme.BLUE)


def test_device_badge_cpu_fallback_is_amber() -> None:
    cuda = DeviceState(checked=True, cuda_detected=True, gpu_name="X")
    info = EngineInfo("small", "cpu", "int8", fallback_reason="falló", cuda_error_detail="d")
    assert device_badge(cuda, info) == ("CPU · int8 (CUDA falló)", theme.WARNING)


def test_engine_badge_states() -> None:
    assert engine_badge(None, loading_model=None, failed=False) == ("Motor sin cargar", theme.MUTED)
    assert engine_badge(None, loading_model="medium", failed=False) == ("Cargando modelo medium…", theme.WARNING)
    info = EngineInfo("medium", "cuda", "int8_float16")
    assert engine_badge(info, loading_model=None, failed=False) == ("Listo · medium", theme.SUCCESS)
    assert engine_badge(None, loading_model=None, failed=True) == ("Error", theme.ERROR)
