import json
from pathlib import Path

from whisper_studio import self_test
from whisper_studio.downloader import ProbeResult
from whisper_studio.models import Segment, TranscriptResult
from whisper_studio.transcriber import EngineInfo


class FakeEngine:
    def ensure_loaded(self, model_name, device_pref, on_status):
        return EngineInfo(model_name, "cuda", "int8_float16")

    def transcribe(self, source, language, word_timestamps, vad_filter, on_segment, cancel_event):
        return TranscriptResult([Segment(0.0, 1.0, "hello")], "en", 0.9, 3.0, "tiny", "cuda", "int8_float16")


def _ok_probe(url: str) -> ProbeResult:
    return ProbeResult(kind="video", id="abc", title="Me at the zoo")


def _failing_probe(url: str) -> ProbeResult:
    raise RuntimeError("offline")


def test_success_writes_report_and_returns_zero(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    code = self_test.run_self_test(tmp_path / "a.wav", report, engine=FakeEngine(), probe_fn=_ok_probe)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert code == 0
    assert data["ok"] is True
    assert data["transcription"]["device"] == "cuda"
    assert data["transcription"]["compute_type"] == "int8_float16"
    assert data["probe"]["title"] == "Me at the zoo"


def test_probe_failure_is_reported_and_returns_one(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    code = self_test.run_self_test(tmp_path / "a.wav", report, engine=FakeEngine(), probe_fn=_failing_probe)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert code == 1
    assert data["ok"] is False
    assert "offline" in data["probe"]["error"]
    assert data["transcription"]["device"] == "cuda"


def test_engine_failure_does_not_skip_probe(tmp_path: Path) -> None:
    class Broken(FakeEngine):
        def ensure_loaded(self, *a, **k):
            raise RuntimeError("no model")

    report = tmp_path / "r.json"
    code = self_test.run_self_test(tmp_path / "a.wav", report, engine=Broken(), probe_fn=_ok_probe)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert code == 1
    assert "no model" in data["transcription"]["error"]
    assert data["probe"]["title"] == "Me at the zoo"


def test_parse_args() -> None:
    assert self_test.parse_args(["--self-test", "a.wav", "r.json"]) == (Path("a.wav"), Path("r.json"))
    assert self_test.parse_args([]) is None
    assert self_test.parse_args(["--self-test", "a.wav"]) is None
