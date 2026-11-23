import gc
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from whisper_studio.errors import CancelledError, EngineError
from whisper_studio.logging_setup import get_logger
from whisper_studio.models import Segment, TranscriptResult, Word
from whisper_studio.paths import data_dir

log = get_logger("transcriber")

_CUDA_PRECISIONS = ("int8_float16", "float16", "int8")
_CPU_PRECISION = "int8"
_PROBE_SAMPLE_RATE = 16000

# After a non-memory CUDA failure (e.g. cuBLAS missing), a second CUDA encode in the same process
# deadlocks inside ctranslate2 instead of raising. The failure is therefore remembered for the
# whole process and CUDA is not attempted again until the app restarts. Out-of-memory errors are
# excluded because retrying with a smaller model is legitimate and does not hang.
_process_cuda_failure: tuple[str, str] | None = None
_RESTART_HINT = " Reinicia la aplicación para volver a intentar usar la GPU."


def _is_out_of_memory(exc: BaseException) -> bool:
    lowered = str(exc).lower()
    return "out of memory" in lowered or "cuda_error_out_of_memory" in lowered

ModelFactory = Callable[[str, str, str], Any]
ModelDownloader = Callable[[str], str]
SupportedComputeTypes = Callable[[str], set[str]]
StatusCallback = Callable[[str], None]
SegmentCallback = Callable[[Segment, float | None], None]


@dataclass(frozen=True)
class EngineInfo:
    model: str
    device: str
    compute_type: str
    fallback_reason: str | None = None
    cuda_error_detail: str | None = None


def classify_engine_error(exc: BaseException) -> str:
    text = str(exc)
    lowered = text.lower()
    if _is_out_of_memory(exc):
        return (
            "La GPU se quedó sin memoria. Prueba con un modelo más pequeño "
            "(por ejemplo small o base) o selecciona el dispositivo CPU."
        )
    if any(token in lowered for token in ("cublas", "cudnn", "is not found or cannot be loaded", ".dll")):
        return (
            "Faltan o no se pueden cargar las bibliotecas NVIDIA (cuBLAS/cuDNN). "
            "Ejecuta «uv sync» para reinstalarlas y comprueba el controlador de la GPU; "
            "mientras tanto puedes usar el dispositivo CPU."
        )
    return f"Error del motor de transcripción: {text or type(exc).__name__}"


def _default_downloader(model_name: str) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from faster_whisper.utils import download_model

    from whisper_studio.cuda_setup import register_nvidia_dll_dirs

    register_nvidia_dll_dirs()
    cache = data_dir() / "models"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        return download_model(model_name, local_files_only=True, cache_dir=str(cache))
    except Exception:
        return download_model(model_name, cache_dir=str(cache))


def _default_factory(path: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(path, device=device, compute_type=compute_type)


def _default_supported(device: str) -> set[str]:
    import ctranslate2

    return set(ctranslate2.get_supported_compute_types(device))


class WhisperEngine:
    """Holds at most one loaded model; all access is serialized by a lock."""

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        model_downloader: ModelDownloader | None = None,
        supported_compute_types: SupportedComputeTypes | None = None,
    ) -> None:
        self._factory = model_factory or _default_factory
        self._downloader = model_downloader or _default_downloader
        self._supported = supported_compute_types or _default_supported
        self._lock = threading.RLock()
        self._model: Any = None
        self._key: tuple[str, str] | None = None
        self._info: EngineInfo | None = None

    @property
    def info(self) -> EngineInfo | None:
        return self._info

    def unload(self) -> None:
        with self._lock:
            self._release()

    def _release(self) -> None:
        # Dropping the reference and collecting is the only way to free GPU memory
        # before a larger model is loaded; two big models never coexist on 4 GB.
        self._model = None
        self._key = None
        self._info = None
        gc.collect()

    def ensure_loaded(
        self, model_name: str, device_pref: str, on_status: StatusCallback
    ) -> EngineInfo:
        if device_pref not in ("auto", "cuda", "cpu"):
            raise EngineError(f"Dispositivo no válido: {device_pref}")
        with self._lock:
            key = (model_name, device_pref)
            if self._model is not None and self._key == key and self._info is not None:
                return self._info
            self._release()
            on_status(f"Preparando el modelo «{model_name}» (la primera vez se descarga)…")
            path = self._resolve_model_path(model_name)
            if device_pref == "cpu":
                info = self._load_cpu(path, model_name, on_status, None, None)
            else:
                info = self._load_cuda_or_fallback(path, model_name, device_pref, on_status)
            self._key = key
            self._info = info
            return info

    def _resolve_model_path(self, model_name: str) -> str:
        try:
            return self._downloader(model_name)
        except Exception as exc:
            log.warning("Model download failed", exc_info=True)
            raise EngineError(
                f"No se pudo descargar el modelo «{model_name}». "
                "Comprueba tu conexión a internet e inténtalo de nuevo.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    def _pick_cuda_precision(self) -> str:
        supported = self._supported("cuda")
        for precision in _CUDA_PRECISIONS:
            if precision in supported:
                return precision
        raise RuntimeError(f"No supported CUDA compute type among {sorted(supported)}")

    def _probe(self, model: Any) -> None:
        # Construction succeeds even when cuBLAS is missing; only a real encode exposes it.
        audio = np.zeros(_PROBE_SAMPLE_RATE, dtype=np.float32)
        segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        for _ in segments:
            pass

    def _load_cuda_or_fallback(
        self, path: str, model_name: str, device_pref: str, on_status: StatusCallback
    ) -> EngineInfo:
        global _process_cuda_failure
        if _process_cuda_failure is not None:
            message, detail = _process_cuda_failure
            if device_pref == "cuda":
                raise EngineError(message + _RESTART_HINT, detail=detail)
            reason = f"No se pudo usar la GPU, se usará la CPU (más lento). {message}"
            return self._load_cpu(path, model_name, on_status, reason, detail)

        on_status("Comprobando la GPU con una transcripción de prueba…")
        model: Any = None
        try:
            precision = self._pick_cuda_precision()
            model = self._factory(path, "cuda", precision)
            self._probe(model)
        except Exception as exc:
            model = None
            gc.collect()
            detail = f"{type(exc).__name__}: {exc}"
            log.warning("CUDA unavailable: %s", detail)
            message = classify_engine_error(exc)
            if not _is_out_of_memory(exc):
                _process_cuda_failure = (message, detail)
            if device_pref == "cuda":
                raise EngineError(message, detail=detail) from exc
            reason = f"No se pudo usar la GPU, se usará la CPU (más lento). {message}"
            return self._load_cpu(path, model_name, on_status, reason, detail)
        self._model = model
        return EngineInfo(model_name, "cuda", precision)

    def _load_cpu(
        self,
        path: str,
        model_name: str,
        on_status: StatusCallback,
        fallback_reason: str | None,
        cuda_detail: str | None,
    ) -> EngineInfo:
        on_status("Cargando el modelo en la CPU…")
        try:
            self._model = self._factory(path, "cpu", _CPU_PRECISION)
        except Exception as exc:
            raise EngineError(
                classify_engine_error(exc), detail=f"{type(exc).__name__}: {exc}"
            ) from exc
        return EngineInfo(model_name, "cpu", _CPU_PRECISION, fallback_reason, cuda_detail)

    def transcribe(
        self,
        source: str | Path | np.ndarray,
        language: str | None,
        word_timestamps: bool,
        vad_filter: bool,
        on_segment: SegmentCallback,
        cancel_event: threading.Event,
    ) -> TranscriptResult:
        with self._lock:
            if self._model is None or self._info is None:
                raise EngineError("El modelo de transcripción no está cargado.")
            if cancel_event.is_set():
                raise CancelledError()
            audio = source if isinstance(source, np.ndarray) else str(source)
            lang = None if language in (None, "auto") else language
            try:
                # Not conditioning on previous text stops one hallucinated line from
                # being fed back and repeated for the rest of the file.
                generator, info = self._model.transcribe(
                    audio,
                    language=lang,
                    beam_size=5,
                    condition_on_previous_text=False,
                    word_timestamps=word_timestamps,
                    vad_filter=vad_filter,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
            except Exception as exc:
                raise self._engine_error(exc) from exc
            return self._consume(generator, info, on_segment, cancel_event)

    def _consume(
        self, generator: Any, info: Any, on_segment: SegmentCallback, cancel_event: threading.Event
    ) -> TranscriptResult:
        assert self._info is not None
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        segments: list[Segment] = []
        try:
            for raw in generator:
                segment = _to_segment(raw)
                segments.append(segment)
                progress = min(1.0, max(0.0, segment.end / duration)) if duration > 0 else None
                on_segment(segment, progress)
                if cancel_event.is_set():
                    raise CancelledError()
        except CancelledError:
            raise
        except Exception as exc:
            raise self._engine_error(exc) from exc
        finally:
            close = getattr(generator, "close", None)
            if close is not None:
                close()
        return TranscriptResult(
            segments=segments,
            language=str(info.language),
            language_probability=float(info.language_probability),
            duration=duration,
            model=self._info.model,
            device=self._info.device,
            compute_type=self._info.compute_type,
        )

    @staticmethod
    def _engine_error(exc: Exception) -> EngineError:
        log.error("Inference failed", exc_info=True)
        return EngineError(classify_engine_error(exc), detail=f"{type(exc).__name__}: {exc}")


def _to_segment(raw: Any) -> Segment:
    words = None
    raw_words = getattr(raw, "words", None)
    if raw_words:
        words = [Word(float(w.start), float(w.end), str(w.word), float(w.probability)) for w in raw_words]
    return Segment(float(raw.start), float(raw.end), str(raw.text), words)
