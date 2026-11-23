import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from whisper_studio.errors import CancelledError, EngineError
from whisper_studio.models import Segment
from whisper_studio import transcriber
from whisper_studio.transcriber import WhisperEngine, classify_engine_error


@pytest.fixture(autouse=True)
def fresh_process_cuda_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcriber, "_process_cuda_failure", None)


def test_classify_oom() -> None:
    for text in ("CUDA failed with error out of memory", "CUDA_ERROR_OUT_OF_MEMORY", "Out Of Memory"):
        msg = classify_engine_error(RuntimeError(text))
        assert "memoria" in msg and "modelo más pequeño" in msg


def test_classify_missing_libraries() -> None:
    for text in (
        "Library cublas64_12.dll is not found or cannot be loaded",
        "Could not load library cudnn_ops64_9.dll",
        "cuDNN error",
    ):
        msg = classify_engine_error(RuntimeError(text))
        assert "NVIDIA" in msg and "uv sync" in msg


def test_classify_generic() -> None:
    msg = classify_engine_error(ValueError("boom"))
    assert "boom" in msg and msg.startswith("Error del motor")


@dataclass
class FakeSeg:
    start: float
    end: float
    text: str
    words: Any = None


@dataclass
class FakeModel:
    segments: list[FakeSeg] = field(default_factory=list)
    duration: float = 10.0
    fail_with: Exception | None = None
    fail_after: int | None = None
    calls: list[dict] = field(default_factory=list)

    def transcribe(self, audio: Any, **kwargs: Any):
        self.calls.append({"audio": audio, **kwargs})
        if self.fail_with is not None and self.fail_after is None:
            raise self.fail_with

        def gen():
            for i, seg in enumerate(self.segments):
                if self.fail_with is not None and i == self.fail_after:
                    raise self.fail_with
                yield seg

        info = SimpleNamespace(language="es", language_probability=0.9, duration=self.duration)
        return gen(), info


class Factory:
    def __init__(self, behavior: dict[str, Any] | None = None) -> None:
        self.behavior = behavior or {}
        self.created: list[tuple[str, str, str]] = []
        self.models: list[FakeModel] = []

    def __call__(self, path: str, device: str, compute_type: str) -> FakeModel:
        self.created.append((path, device, compute_type))
        action = self.behavior.get(device)
        if isinstance(action, Exception):
            raise action
        model = action() if callable(action) else FakeModel()
        self.models.append(model)
        return model


def make_engine(
    factory: Factory, supported: set[str] | Exception | None = None, download: Any = None
) -> WhisperEngine:
    def compute_types(device: str) -> set[str]:
        if isinstance(supported, Exception):
            raise supported
        return supported if supported is not None else {"int8", "int8_float16", "float16"}

    return WhisperEngine(
        model_factory=factory,
        model_downloader=download or (lambda name: f"/models/{name}"),
        supported_compute_types=compute_types,
    )


def test_auto_uses_cuda_int8_float16_and_probes() -> None:
    factory = Factory()
    info = make_engine(factory).ensure_loaded("small", "auto", lambda s: None)
    assert (info.model, info.device, info.compute_type) == ("small", "cuda", "int8_float16")
    assert info.fallback_reason is None
    assert factory.models[0].calls, "a real probe inference must run"
    audio = factory.models[0].calls[0]["audio"]
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32 and audio.shape == (16000,)


def test_precision_preference_order() -> None:
    info = make_engine(Factory(), supported={"float16", "int8"}).ensure_loaded("tiny", "cuda", print)
    assert info.compute_type == "float16"
    info = make_engine(Factory(), supported={"int8"}).ensure_loaded("tiny", "cuda", print)
    assert info.compute_type == "int8"


def test_auto_falls_back_to_cpu_when_load_fails() -> None:
    factory = Factory({"cuda": RuntimeError("Library cublas64_12.dll is not found")})
    info = make_engine(factory).ensure_loaded("medium", "auto", lambda s: None)
    assert (info.model, info.device, info.compute_type) == ("medium", "cpu", "int8")
    assert info.fallback_reason and "CPU" in info.fallback_reason
    assert "cublas64_12.dll" in (info.cuda_error_detail or "")


def test_auto_falls_back_when_probe_inference_fails() -> None:
    def broken() -> FakeModel:
        return FakeModel(fail_with=RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))

    factory = Factory({"cuda": broken})
    info = make_engine(factory).ensure_loaded("small", "auto", lambda s: None)
    assert info.device == "cpu"
    assert info.fallback_reason
    assert [c[1] for c in factory.created] == ["cuda", "cpu"]


def test_auto_falls_back_without_cuda_devices() -> None:
    info = make_engine(Factory(), supported=RuntimeError("no cuda")).ensure_loaded("small", "auto", print)
    assert info.device == "cpu" and info.fallback_reason


def test_cuda_preference_does_not_fall_back() -> None:
    factory = Factory({"cuda": RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")})
    with pytest.raises(EngineError) as exc:
        make_engine(factory).ensure_loaded("large-v3", "cuda", print)
    assert "memoria" in exc.value.user_message
    assert "CUDA_ERROR_OUT_OF_MEMORY" in exc.value.detail
    assert all(c[1] == "cuda" for c in factory.created)


def test_library_failure_is_not_retried_in_same_process() -> None:
    def broken() -> FakeModel:
        return FakeModel(fail_with=RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))

    factory = Factory({"cuda": broken})
    engine = make_engine(factory)
    engine.ensure_loaded("tiny", "auto", lambda s: None)
    info = engine.ensure_loaded("base", "auto", lambda s: None)
    assert info.device == "cpu" and info.fallback_reason
    assert [c[1] for c in factory.created] == ["cuda", "cpu", "cpu"]

    other = make_engine(factory)
    with pytest.raises(EngineError) as exc:
        other.ensure_loaded("tiny", "cuda", lambda s: None)
    assert "Reinicia" in exc.value.user_message
    assert [c[1] for c in factory.created].count("cuda") == 1


def test_out_of_memory_does_not_block_later_cuda_attempts() -> None:
    attempts = iter([RuntimeError("CUDA failed with error out of memory"), None])

    def model() -> FakeModel:
        return FakeModel(fail_with=next(attempts))

    factory = Factory({"cuda": model})
    engine = make_engine(factory)
    assert engine.ensure_loaded("large-v3", "auto", lambda s: None).device == "cpu"
    assert engine.ensure_loaded("small", "auto", lambda s: None).device == "cuda"


def test_cpu_uses_int8_without_probe() -> None:
    factory = Factory()
    info = make_engine(factory).ensure_loaded("base", "cpu", print)
    assert (info.device, info.compute_type, info.fallback_reason) == ("cpu", "int8", None)
    assert factory.models[0].calls == []


def test_reuse_and_swap_releases_previous() -> None:
    factory = Factory()
    engine = make_engine(factory)
    a = engine.ensure_loaded("small", "cpu", print)
    b = engine.ensure_loaded("small", "cpu", print)
    assert a == b and len(factory.created) == 1
    engine.ensure_loaded("base", "cpu", print)
    assert len(factory.created) == 2
    engine.unload()
    with pytest.raises(EngineError):
        engine.transcribe(np.zeros(10, np.float32), "es", False, True, lambda s, p: None, threading.Event())


def test_download_failure_is_reported_in_spanish() -> None:
    def failing(name: str) -> str:
        raise OSError("connection refused")

    with pytest.raises(EngineError) as exc:
        make_engine(Factory(), download=failing).ensure_loaded("small", "auto", print)
    assert exc.value.user_message.startswith("No se pudo descargar el modelo")
    assert "connection refused" in exc.value.detail


def loaded_engine(model_kwargs: dict[str, Any]) -> tuple[WhisperEngine, Factory]:
    def build() -> FakeModel:
        return FakeModel(**model_kwargs)

    factory = Factory({"cpu": build})
    engine = make_engine(factory)
    engine.ensure_loaded("small", "cpu", print)
    return engine, factory


def test_transcribe_emits_segments_with_real_progress() -> None:
    words = [SimpleNamespace(start=0.0, end=1.0, word=" Hola", probability=0.9)]
    engine, factory = loaded_engine(
        {"segments": [FakeSeg(0, 2.5, " Hola ", words), FakeSeg(2.5, 5.0, "mundo")], "duration": 10.0}
    )
    seen: list[tuple[Segment, float | None]] = []
    result = engine.transcribe(
        "file.wav", "auto", True, True, lambda s, p: seen.append((s, p)), threading.Event()
    )
    call = factory.models[0].calls[0]
    assert call["language"] is None
    assert call["beam_size"] == 5
    assert call["condition_on_previous_text"] is False
    assert call["word_timestamps"] is True
    assert call["vad_filter"] is True
    assert call["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert [p for _, p in seen] == [0.25, 0.5]
    assert seen[0][0].words and seen[0][0].words[0].word == " Hola"
    assert seen[1][0].words is None
    assert result.language == "es" and result.duration == 10.0
    assert (result.model, result.device, result.compute_type) == ("small", "cpu", "int8")
    assert [s.text for s in result.segments] == [" Hola ", "mundo"]


def test_progress_is_none_without_duration_and_clamped() -> None:
    engine, _ = loaded_engine({"segments": [FakeSeg(0, 2, "a")], "duration": 0.0})
    seen: list[float | None] = []
    engine.transcribe("f", "es", False, False, lambda s, p: seen.append(p), threading.Event())
    assert seen == [None]
    engine, _ = loaded_engine({"segments": [FakeSeg(0, 12, "a")], "duration": 10.0})
    seen.clear()
    engine.transcribe("f", "es", False, False, lambda s, p: seen.append(p), threading.Event())
    assert seen == [1.0]


def test_cancel_between_segments() -> None:
    engine, _ = loaded_engine({"segments": [FakeSeg(i, i + 1, f"s{i}") for i in range(5)]})
    cancel = threading.Event()
    seen: list[Segment] = []

    def on_segment(seg: Segment, progress: float | None) -> None:
        seen.append(seg)
        if len(seen) == 2:
            cancel.set()

    with pytest.raises(CancelledError):
        engine.transcribe("f", "es", False, False, on_segment, cancel)
    assert len(seen) == 2


def test_cancel_before_start() -> None:
    engine, factory = loaded_engine({"segments": [FakeSeg(0, 1, "a")]})
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        engine.transcribe("f", "es", False, False, lambda s, p: None, cancel)


def test_oom_during_inference_is_classified() -> None:
    engine, _ = loaded_engine(
        {
            "segments": [FakeSeg(0, 1, "a"), FakeSeg(1, 2, "b")],
            "fail_with": RuntimeError("CUDA failed with error out of memory"),
            "fail_after": 1,
        }
    )
    with pytest.raises(EngineError) as exc:
        engine.transcribe("f", "es", False, False, lambda s, p: None, threading.Event())
    assert "memoria" in exc.value.user_message
    assert "out of memory" in exc.value.detail


def test_transcribe_accepts_numpy_array() -> None:
    engine, factory = loaded_engine({"segments": []})
    audio = np.zeros(16000, dtype=np.float32)
    engine.transcribe(audio, "en", False, True, lambda s, p: None, threading.Event())
    assert factory.models[0].calls[0]["audio"] is audio
    assert factory.models[0].calls[0]["language"] == "en"
