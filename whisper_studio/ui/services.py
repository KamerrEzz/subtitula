"""Work that must never run on the Tk thread, plus the bridge that hands results back to it."""

import io
import os
import queue
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps

from whisper_studio.cuda_setup import register_nvidia_dll_dirs
from whisper_studio.logging_setup import get_logger
from whisper_studio.ui.helpers import DeviceState

log = get_logger("ui")

THUMB_SIZE = (160, 90)
_THUMB_SCALE = 2
_THUMB_TIMEOUT_S = 15
_THUMB_MAX_BYTES = 5 * 1024 * 1024


class UiBridge:
    """Threads post callables here; the Tk thread drains and runs them from its poll loop."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue()

    def post(self, call: Callable[[], None]) -> None:
        self._queue.put(call)

    def run_async(self, work: Callable[[], Any], on_done: Callable[[Any, BaseException | None], None]) -> None:
        def target() -> None:
            try:
                result, error = work(), None
            except BaseException as exc:
                log.warning("Background task failed", exc_info=True)
                result, error = None, exc
            self.post(lambda: on_done(result, error))

        threading.Thread(target=target, name="ui-worker", daemon=True).start()

    def drain(self, limit: int = 50) -> None:
        for _ in range(limit):
            try:
                call = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                call()
            except Exception:
                log.exception("UI callback failed")


def _nvidia_smi_gpu_name() -> str:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8, creationflags=flags, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    lines = proc.stdout.strip().splitlines()
    return lines[0].strip() if lines else ""


def detect_device() -> DeviceState:
    try:
        register_nvidia_dll_dirs()
        import ctranslate2

        count = int(ctranslate2.get_cuda_device_count())
    except Exception:
        log.warning("CUDA detection failed", exc_info=True)
        return DeviceState(checked=True)
    if count <= 0:
        return DeviceState(checked=True)
    return DeviceState(checked=True, cuda_detected=True, gpu_name=_nvidia_smi_gpu_name())


def _round_corners(image: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *image.size), radius=radius, fill=255)
    rounded = image.convert("RGBA")
    rounded.putalpha(mask)
    return rounded


def fetch_thumbnail(url: str) -> Image.Image:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=_THUMB_TIMEOUT_S) as response:
        data = response.read(_THUMB_MAX_BYTES)
    image = Image.open(io.BytesIO(data))
    target = (THUMB_SIZE[0] * _THUMB_SCALE, THUMB_SIZE[1] * _THUMB_SCALE)
    return _round_corners(ImageOps.fit(image.convert("RGB"), target), 12 * _THUMB_SCALE)


def make_app_icon(size: int = 256) -> Image.Image:
    """Violet-to-blue rounded square with a white waveform."""
    gradient = Image.new("RGB", (size, size))
    start, end = (0x7C, 0x5C, 0xFF), (0x3B, 0x82, 0xF6)
    pixels = gradient.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            pixels[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(start, end))
    icon = _round_corners(gradient, size // 4)
    draw = ImageDraw.Draw(icon)
    heights = (0.22, 0.42, 0.62, 0.34, 0.5, 0.26)
    bar_w = size * 0.075
    gap = size * 0.06
    total = len(heights) * bar_w + (len(heights) - 1) * gap
    x = (size - total) / 2
    for h in heights:
        half = size * h / 2
        draw.rounded_rectangle((x, size / 2 - half, x + bar_w, size / 2 + half), radius=bar_w / 2, fill="white")
        x += bar_w + gap
    return icon


def open_path(path: Path) -> None:
    os.startfile(str(path))  # type: ignore[attr-defined]  # Windows-only API


def read_transcript_file(output_dir: Path | None) -> str | None:
    """None means the file is missing; an empty string means the transcript is genuinely empty."""
    if output_dir is None:
        return None
    try:
        return (output_dir / "transcripcion.txt").read_text(encoding="utf-8")
    except OSError:
        return None

