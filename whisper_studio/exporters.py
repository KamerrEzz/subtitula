import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from whisper_studio.models import Segment, TranscriptResult
from whisper_studio.paths import atomic_write_text
from whisper_studio.subtitles import build_cues

PARAGRAPH_GAP_S = 2.0
PARAGRAPH_SEGMENTS = 6
_SENTENCE_END = (".", "!", "?", "…", "。")

FILENAMES = {
    "txt": "transcripcion.txt",
    "srt": "transcripcion.srt",
    "vtt": "transcripcion.vtt",
    "md": "transcripcion.md",
    "json": "metadata.json",
}


@dataclass(frozen=True)
class ExportMeta:
    title: str
    source_url: str | None
    duration: float | None
    language: str
    model: str
    device: str
    created_at: datetime


def _split_ms(seconds: float) -> tuple[int, int, int, int]:
    # Rounding the total first avoids "00:00:59,1000" when the fraction rounds up to 1 s.
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return hours, minutes, secs, ms


def format_srt_time(seconds: float) -> str:
    h, m, s, ms = _split_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    h, m, s, ms = _split_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_clock(seconds: float, with_hours: bool = True) -> str:
    h, m, s, _ = _split_ms(seconds)
    if with_hours:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{h * 60 + m:02d}:{s:02d}"


def _spoken(result: TranscriptResult) -> list[Segment]:
    return [s for s in result.segments if s.text.strip()]


def to_txt(result: TranscriptResult) -> str:
    paragraphs: list[list[str]] = []
    current: list[str] = []
    prev_end: float | None = None
    for seg in _spoken(result):
        text = seg.text.strip()
        gap = prev_end is not None and seg.start - prev_end > PARAGRAPH_GAP_S
        full = len(current) >= PARAGRAPH_SEGMENTS and current[-1].endswith(_SENTENCE_END)
        if current and (gap or full):
            paragraphs.append(current)
            current = []
        current.append(text)
        prev_end = seg.end
    if current:
        paragraphs.append(current)
    if not paragraphs:
        return ""
    return "\n\n".join(" ".join(p) for p in paragraphs) + "\n"


def _cues(result: TranscriptResult, readable: bool) -> list[tuple[float, float, str]]:
    if readable:
        return [(c.start, c.end, "\n".join(c.lines)) for c in build_cues(result.segments)]
    return [(s.start, s.end, s.text.strip()) for s in _spoken(result)]


def to_srt(result: TranscriptResult, *, readable: bool = False) -> str:
    blocks = [
        f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n"
        for i, (start, end, text) in enumerate(_cues(result, readable), 1)
    ]
    return "\n".join(blocks)


def to_vtt(result: TranscriptResult, *, readable: bool = False) -> str:
    blocks = [
        f"{format_vtt_time(start)} --> {format_vtt_time(end)}\n{text}\n"
        for start, end, text in _cues(result, readable)
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_markdown(result: TranscriptResult, meta: ExportMeta) -> str:
    lines = [f"# {meta.title}", ""]
    if meta.source_url:
        lines.append(f"- URL: {meta.source_url}")
    if meta.duration is not None:
        lines.append(f"- Duración: {format_clock(meta.duration)}")
    lines.append(f"- Idioma: {meta.language}")
    lines.append(f"- Modelo: {meta.model}")
    lines.append(f"- Dispositivo: {meta.device}")
    lines += ["", "## Transcripción", ""]
    segments = _spoken(result)
    reference = max([result.duration, *(s.end for s in segments)], default=0.0)
    long_media = reference >= 3600
    for seg in segments:
        stamp = format_clock(seg.start, with_hours=long_media)
        lines.append(f"**[{stamp}]** {seg.text.strip()}")
        lines.append("")
    return "\n".join(lines)


def _segment_dict(seg: Segment) -> dict[str, Any]:
    data: dict[str, Any] = {
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": seg.text.strip(),
    }
    if seg.words:
        data["words"] = [
            {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
            for w in seg.words
        ]
    return data


def to_json(
    result: TranscriptResult,
    meta: ExportMeta,
    include_segments: bool = True,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "title": meta.title,
        "source_url": meta.source_url,
        "duration": meta.duration,
        "language": result.language or meta.language,
        "language_probability": result.language_probability,
        "model": result.model or meta.model,
        "device": result.device or meta.device,
        "compute_type": result.compute_type,
        "created_at": meta.created_at.isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    if include_segments:
        payload["segments"] = [_segment_dict(s) for s in _spoken(result)]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_all(
    result: TranscriptResult,
    meta: ExportMeta,
    out_dir: Path,
    formats: list[str],
    readable_subtitles: bool = False,
) -> dict[str, Path]:
    renderers: dict[str, Callable[[], str]] = {
        "txt": lambda: to_txt(result),
        "srt": lambda: to_srt(result, readable=readable_subtitles),
        "vtt": lambda: to_vtt(result, readable=readable_subtitles),
        "md": lambda: to_markdown(result, meta),
        "json": lambda: to_json(result, meta),
    }
    written: dict[str, Path] = {}
    for fmt in formats:
        render = renderers.get(fmt)
        if render is None:
            continue
        target = out_dir / FILENAMES[fmt]
        atomic_write_text(target, render())
        written[fmt] = target
    return written
