from pathlib import Path

import pytest

from whisper_studio import diagnostics
from whisper_studio.config import AppConfig
from whisper_studio.diagnostics import CheckResult, run_diagnostics
from whisper_studio.errors import EngineError
from whisper_studio.transcriber import EngineInfo

EXPECTED_NAMES = [
    "Python",
    "FFmpeg",
    "FFprobe",
    "faster-whisper",
    "GPU CUDA (ctranslate2)",
    "cuBLAS",
    "cuDNN",
    "Inferencia real en GPU",
    "Carpeta de salida",
    "Carpeta de modelos",
]


def by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setattr(diagnostics, "data_dir", lambda: tmp_path / "data")
    return AppConfig(output_dir=str(tmp_path / "out"))


def test_returns_every_check_in_order(cfg: AppConfig) -> None:
    results = run_diagnostics(cfg, deep_cuda=False)
    assert [r.name for r in results] == EXPECTED_NAMES
    assert all(r.ok in (True, False, None) and r.detail for r in results)


def test_deep_check_is_skipped_by_default(cfg: AppConfig) -> None:
    r = by_name(run_diagnostics(cfg, deep_cuda=False))["Inferencia real en GPU"]
    assert r.ok is None and "no se ejecutó" in r.detail.lower()


def test_missing_ffmpeg_and_ffprobe_give_actionable_fix(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "which_ffmpeg", lambda: None)
    monkeypatch.setattr(diagnostics, "which_ffprobe", lambda: None)
    res = by_name(run_diagnostics(cfg, deep_cuda=False))
    assert res["FFmpeg"].ok is None and "winget install Gyan.FFmpeg" in res["FFmpeg"].fix
    assert res["FFprobe"].ok is None and "winget install Gyan.FFmpeg" in res["FFprobe"].fix


def test_output_dir_writable_and_created(cfg: AppConfig, tmp_path: Path) -> None:
    r = by_name(run_diagnostics(cfg, deep_cuda=False))["Carpeta de salida"]
    assert r.ok is True
    assert list((tmp_path / "out").iterdir()) == []


def test_output_dir_not_writable(cfg: AppConfig, tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    cfg.output_dir = str(blocker / "sub")
    r = by_name(run_diagnostics(cfg, deep_cuda=False))["Carpeta de salida"]
    assert r.ok is False and r.fix


def test_models_dir_low_space_warns(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "check_free_space", lambda p, n: False)
    r = by_name(run_diagnostics(cfg, deep_cuda=False))["Carpeta de modelos"]
    assert r.ok is None and "espacio" in r.detail.lower()


def test_deep_check_success(cfg: AppConfig) -> None:
    class Engine:
        def ensure_loaded(self, model, device, on_status):
            assert (model, device) == ("tiny", "cuda")
            return EngineInfo("tiny", "cuda", "int8_float16")

        def unload(self) -> None:
            raise AssertionError("an injected engine belongs to the caller")

    engine = Engine()
    r = by_name(run_diagnostics(cfg, deep_cuda=True, engine=engine))["Inferencia real en GPU"]
    assert r.ok is True and "int8_float16" in r.detail


def test_deep_check_failure(cfg: AppConfig) -> None:
    class Engine:
        def ensure_loaded(self, model, device, on_status):
            raise EngineError("Faltan bibliotecas NVIDIA", "RuntimeError: cublas64_12.dll")

        def unload(self) -> None:
            pass

    r = by_name(run_diagnostics(cfg, deep_cuda=True, engine=Engine()))["Inferencia real en GPU"]
    assert r.ok is False
    assert "Faltan bibliotecas NVIDIA" in r.detail and r.fix


def test_a_crashing_check_does_not_break_the_report(cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str:
        raise RuntimeError("kaput")

    monkeypatch.setattr(diagnostics, "which_ffmpeg", boom)
    res = by_name(run_diagnostics(cfg, deep_cuda=False))
    assert res["FFmpeg"].ok is False and "kaput" in res["FFmpeg"].detail
    assert len(res) == len(EXPECTED_NAMES)
