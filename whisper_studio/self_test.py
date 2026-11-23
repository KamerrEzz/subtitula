"""Headless end-to-end check of the real transcription and download paths, for support use."""

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from whisper_studio import __version__
from whisper_studio.paths import atomic_write_text, is_frozen

PROBE_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
FLAG = "--self-test"


def parse_args(argv: Sequence[str]) -> tuple[Path, Path] | None:
    if len(argv) == 3 and argv[0] == FLAG:
        return Path(argv[1]), Path(argv[2])
    return None


def _default_engine() -> Any:
    from whisper_studio.transcriber import WhisperEngine

    return WhisperEngine()


def _default_probe(url: str) -> Any:
    from whisper_studio.downloader import probe

    return probe(url)


def _check_transcription(engine: Any, wav_path: Path) -> dict[str, Any]:
    try:
        info = engine.ensure_loaded("tiny", "auto", lambda _msg: None)
        result = engine.transcribe(wav_path, None, False, False, lambda *_: None, threading.Event())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "model": info.model,
        "device": result.device,
        "compute_type": result.compute_type,
        "language": result.language,
        "text": " ".join(s.text.strip() for s in result.segments),
        "fallback_reason": info.fallback_reason,
    }


def _check_probe(probe_fn: Callable[[str], Any]) -> dict[str, Any]:
    try:
        result = probe_fn(PROBE_URL)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"id": result.id, "title": result.title}


def run_self_test(
    wav_path: Path,
    report_path: Path,
    engine: Any | None = None,
    probe_fn: Callable[[str], Any] | None = None,
) -> int:
    transcription = _check_transcription(engine or _default_engine(), wav_path)
    probe = _check_probe(probe_fn or _default_probe)
    ok = "error" not in transcription and "error" not in probe
    report = {
        "ok": ok,
        "version": __version__,
        "frozen": is_frozen(),
        "transcription": transcription,
        "probe": probe,
    }
    atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1
