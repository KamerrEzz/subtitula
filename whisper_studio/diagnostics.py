import os
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Protocol

from whisper_studio.config import AppConfig
from whisper_studio.cuda_setup import register_nvidia_dll_dirs
from whisper_studio.errors import EngineError
from whisper_studio.media import check_free_space, which_ffmpeg, which_ffprobe
from whisper_studio.paths import data_dir
from whisper_studio.transcriber import EngineInfo, WhisperEngine

MIN_PYTHON = (3, 12)
MIN_MODELS_FREE_BYTES = 4 * 1024**3
_FFMPEG_FIX = "Instala FFmpeg con: winget install Gyan.FFmpeg (y reinicia la aplicación)."
_UV_SYNC_FIX = "Ejecuta «uv sync» en la carpeta de la aplicación para reinstalar las dependencias."
_DRIVER_FIX = "Actualiza el controlador de NVIDIA desde https://www.nvidia.com/Download/index.aspx"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool | None  # None means warning: the app still works with reduced capability
    detail: str
    fix: str = ""


class _Engine(Protocol):
    def ensure_loaded(
        self, model_name: str, device_pref: str, on_status: Callable[[str], None]
    ) -> EngineInfo: ...

    def unload(self) -> None: ...


def _find_dll(name: str, extra_dirs: list[Path]) -> Path | None:
    search = [*extra_dirs, *(Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p)]
    for directory in search:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _check_python() -> CheckResult:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] >= MIN_PYTHON:
        return CheckResult("Python", True, f"Python {version}")
    return CheckResult(
        "Python", False, f"Python {version}; se requiere 3.12 o superior",
        "Ejecuta la aplicación con «uv run python main.py» para usar el entorno correcto.",
    )


def _check_ffmpeg() -> CheckResult:
    path = which_ffmpeg()
    if path:
        return CheckResult("FFmpeg", True, path)
    return CheckResult(
        "FFmpeg", None,
        "No encontrado. Se podrá transcribir audio, pero la descarga de video usará un único archivo y de menor calidad.",
        _FFMPEG_FIX,
    )


def _check_ffprobe() -> CheckResult:
    path = which_ffprobe()
    if path:
        return CheckResult("FFprobe", True, path)
    return CheckResult(
        "FFprobe", None,
        "No encontrado. La duración de los archivos locales será desconocida y el progreso indeterminado.",
        _FFMPEG_FIX,
    )


def _check_faster_whisper() -> CheckResult:
    try:
        register_nvidia_dll_dirs()
        import faster_whisper  # noqa: F401

        version = metadata.version("faster-whisper")
    except Exception as exc:
        return CheckResult("faster-whisper", False, f"No se pudo importar: {exc}", _UV_SYNC_FIX)
    return CheckResult("faster-whisper", True, f"Versión {version}")


def _cuda_device_count() -> int:
    register_nvidia_dll_dirs()
    import ctranslate2

    return int(ctranslate2.get_cuda_device_count())


def _check_cuda_devices() -> CheckResult:
    name = "GPU CUDA (ctranslate2)"
    try:
        count = _cuda_device_count()
    except Exception as exc:
        return CheckResult(name, None, f"No se pudo consultar la GPU: {exc}", _DRIVER_FIX)
    if count > 0:
        return CheckResult(name, True, f"{count} dispositivo(s) CUDA detectado(s)")
    return CheckResult(
        name, None, "No se detectó ninguna GPU CUDA; se usará la CPU (más lento).", _DRIVER_FIX
    )


def _check_dll(name: str, dll: str, dirs: list[Path], fix: str) -> CheckResult:
    if sys.platform != "win32":
        return CheckResult(name, True, "No aplica en este sistema")
    found = _find_dll(dll, dirs)
    if found:
        return CheckResult(name, True, str(found))
    return CheckResult(name, None, f"No se encontró {dll}; la GPU no funcionará.", fix)


def _check_deep_cuda(engine: _Engine | None) -> CheckResult:
    name = "Inferencia real en GPU"
    owned = engine is None
    engine = engine or WhisperEngine()
    try:
        info = engine.ensure_loaded("tiny", "cuda", lambda _: None)
        return CheckResult(
            name, True, f"Transcripción de prueba correcta en {info.device} ({info.compute_type})"
        )
    except EngineError as exc:
        return CheckResult(
            name, False, f"{exc.user_message} ({exc.detail})" if exc.detail else exc.user_message,
            "Revisa las comprobaciones de cuBLAS/cuDNN y el controlador, o usa el dispositivo CPU.",
        )
    finally:
        if owned:
            engine.unload()


def _check_output_dir(config: AppConfig) -> CheckResult:
    name = "Carpeta de salida"
    path = Path(config.output_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write-test-", delete=True):
            pass
    except OSError as exc:
        return CheckResult(
            name, False, f"No se puede escribir en {path}: {exc}",
            "Elige otra carpeta de salida con permisos de escritura.",
        )
    return CheckResult(name, True, f"{path} (escritura correcta)")


def _check_models_dir() -> CheckResult:
    name = "Carpeta de modelos"
    path = data_dir() / "models"
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write-test-", delete=True):
            pass
    except OSError as exc:
        return CheckResult(
            name, False, f"No se puede escribir en {path}: {exc}",
            "Comprueba los permisos de la carpeta de la aplicación.",
        )
    if not check_free_space(path, MIN_MODELS_FREE_BYTES):
        return CheckResult(
            name, None, f"{path}: queda poco espacio libre (menos de 4 GB); los modelos grandes pueden no caber.",
            "Libera espacio en el disco.",
        )
    return CheckResult(name, True, f"{path} (escritura correcta, espacio suficiente)")


def run_diagnostics(
    config: AppConfig, deep_cuda: bool = False, engine: _Engine | None = None
) -> list[CheckResult]:
    dll_dirs = register_nvidia_dll_dirs()
    checks: list[tuple[str, Callable[[], CheckResult]]] = [
        ("Python", _check_python),
        ("FFmpeg", _check_ffmpeg),
        ("FFprobe", _check_ffprobe),
        ("faster-whisper", _check_faster_whisper),
        ("GPU CUDA (ctranslate2)", _check_cuda_devices),
        ("cuBLAS", lambda: _check_dll("cuBLAS", "cublas64_12.dll", dll_dirs, _UV_SYNC_FIX)),
        ("cuDNN", lambda: _check_dll("cuDNN", "cudnn64_9.dll", dll_dirs, _UV_SYNC_FIX)),
        (
            "Inferencia real en GPU",
            (lambda: _check_deep_cuda(engine))
            if deep_cuda
            else (
                lambda: CheckResult(
                    "Inferencia real en GPU", None,
                    "No se ejecutó (usa la comprobación profunda; descarga el modelo tiny, ~75 MB, la primera vez).",
                )
            ),
        ),
        ("Carpeta de salida", lambda: _check_output_dir(config)),
        ("Carpeta de modelos", _check_models_dir),
    ]
    results: list[CheckResult] = []
    for name, check in checks:
        try:
            results.append(check())
        except Exception as exc:
            results.append(CheckResult(name, False, f"La comprobación falló: {exc}"))
    return results
