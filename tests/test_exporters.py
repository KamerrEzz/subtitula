import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from whisper_studio import exporters
from whisper_studio.exporters import ExportMeta
from whisper_studio.models import Segment, TranscriptResult, Word


def make_result(segments: list[Segment], duration: float = 100.0) -> TranscriptResult:
    return TranscriptResult(
        segments=segments, language="es", language_probability=0.98,
        duration=duration, model="small", device="cuda", compute_type="int8_float16",
    )


META = ExportMeta(
    title="Mi video", source_url="https://youtu.be/abc", duration=3725.0, language="es",
    model="small", device="cuda", created_at=datetime(2026, 1, 2, 3, 4, 5),
)


@pytest.mark.parametrize(
    "seconds,srt,vtt",
    [
        (0.0, "00:00:00,000", "00:00:00.000"),
        (1.5, "00:00:01,500", "00:00:01.500"),
        (3725.123, "01:02:05,123", "01:02:05.123"),
        (59.9996, "00:01:00,000", "00:01:00.000"),
        (3599.9999, "01:00:00,000", "01:00:00.000"),
        (-0.2, "00:00:00,000", "00:00:00.000"),
        (360000.0, "100:00:00,000", "100:00:00.000"),
    ],
)
def test_timestamp_formats(seconds: float, srt: str, vtt: str) -> None:
    assert exporters.format_srt_time(seconds) == srt
    assert exporters.format_vtt_time(seconds) == vtt


def test_srt_structure() -> None:
    result = make_result([Segment(0.0, 1.2346, " Hola "), Segment(61.0, 62.0, "Mundo")])
    out = exporters.to_srt(result)
    assert out == (
        "1\n00:00:00,000 --> 00:00:01,235\nHola\n\n2\n00:01:01,000 --> 00:01:02,000\nMundo\n"
    )
    blocks = out.strip().split("\n\n")
    for i, block in enumerate(blocks, 1):
        lines = block.split("\n")
        assert lines[0] == str(i)
        assert re.fullmatch(r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}", lines[1])


def test_vtt_structure() -> None:
    result = make_result([Segment(0.0, 1.0, "Hola")])
    out = exporters.to_vtt(result)
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.000\nHola" in out
    assert "," not in out.split("\n")[2]


def test_empty_segments_are_skipped() -> None:
    result = make_result([Segment(0, 1, "  "), Segment(1, 2, "x")])
    assert exporters.to_srt(result).startswith("1\n00:00:01,000")
    assert exporters.to_txt(result) == "x\n"


def test_txt_paragraph_on_gap() -> None:
    result = make_result([Segment(0, 1, "Uno."), Segment(1.5, 2, "Dos."), Segment(5.0, 6, "Tres.")])
    assert exporters.to_txt(result) == "Uno. Dos.\n\nTres.\n"


def test_txt_paragraph_every_six_sentences() -> None:
    segs = [Segment(i, i + 0.9, f"Frase {i}.") for i in range(7)]
    out = exporters.to_txt(make_result(segs))
    paragraphs = out.strip().split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0].count("Frase") == 6


def test_txt_no_split_without_punctuation() -> None:
    segs = [Segment(i, i + 0.9, f"palabra{i}") for i in range(8)]
    assert "\n\n" not in exporters.to_txt(make_result(segs))


def test_txt_empty() -> None:
    assert exporters.to_txt(make_result([])) == ""


def test_markdown_fields_and_short_timestamps() -> None:
    meta = ExportMeta("Mi video", None, 125.0, "es", "small", "cuda", datetime(2026, 1, 1))
    result = make_result([Segment(5.0, 6.0, "Hola"), Segment(65.0, 66.0, "Adios")], duration=125.0)
    md = exporters.to_markdown(result, meta)
    assert md.startswith("# Mi video\n")
    assert "URL" not in md
    assert "Duración: 00:02:05" in md
    assert "Idioma: es" in md
    assert "Modelo: small" in md
    assert "Dispositivo: cuda" in md
    assert "## Transcripción" in md
    assert "**[00:05]** Hola" in md
    assert "**[01:05]** Adios" in md


def test_markdown_long_uses_hours_and_url() -> None:
    result = make_result([Segment(3700.0, 3701.0, "tarde")], duration=3725.0)
    md = exporters.to_markdown(result, META)
    assert "URL: https://youtu.be/abc" in md
    assert "**[01:01:40]** tarde" in md


def test_json_roundtrip_with_words() -> None:
    words = [Word(0.0, 0.5, "Hola", 0.9)]
    result = make_result([Segment(0.0, 0.5, "Hola", words), Segment(1.0, 2.0, "x")])
    data = json.loads(exporters.to_json(result, META))
    assert data["title"] == "Mi video"
    assert data["source_url"] == "https://youtu.be/abc"
    assert data["language"] == "es"
    assert data["model"] == "small"
    assert data["device"] == "cuda"
    assert data["compute_type"] == "int8_float16"
    assert data["created_at"] == "2026-01-02T03:04:05"
    assert data["segments"][0]["words"] == [
        {"start": 0.0, "end": 0.5, "word": "Hola", "probability": 0.9}
    ]
    assert "words" not in data["segments"][1]


def test_json_keeps_unicode_and_options() -> None:
    result = make_result([Segment(0, 1, "canción")])
    raw = exporters.to_json(result, META, include_segments=False, extra={"status": "completed"})
    assert "canción" not in raw
    data = json.loads(raw)
    assert data["status"] == "completed"
    assert "segments" not in data
    raw2 = exporters.to_json(result, META)
    assert "canción" in raw2


def test_export_all_writes_files(tmp_path: Path) -> None:
    result = make_result([Segment(0, 1, "Hola.")])
    out = exporters.export_all(result, META, tmp_path / "o", ["txt", "srt", "vtt", "md", "json"])
    assert {k: v.name for k, v in out.items()} == {
        "txt": "transcripcion.txt", "srt": "transcripcion.srt", "vtt": "transcripcion.vtt",
        "md": "transcripcion.md", "json": "metadata.json",
    }
    assert all(p.exists() for p in out.values())
    assert (tmp_path / "o" / "transcripcion.txt").read_text(encoding="utf-8") == "Hola.\n"


def test_export_all_subset_and_unknown(tmp_path: Path) -> None:
    result = make_result([Segment(0, 1, "Hola.")])
    out = exporters.export_all(result, META, tmp_path, ["srt", "pdf"])
    assert list(out) == ["srt"]
    assert not (tmp_path / "transcripcion.txt").exists()


def _long_words_result() -> TranscriptResult:
    words = [Word(i * 0.3, i * 0.3 + 0.3, " palabra" if i % 5 else " Hola,", 0.9) for i in range(30)]
    text = "".join(w.word for w in words)
    return make_result([Segment(0.0, 9.0, text, words)])


def test_srt_readable_splits_long_segments_into_short_cues() -> None:
    out = exporters.to_srt(_long_words_result(), readable=True)
    blocks = out.strip().split("\n\n")
    assert len(blocks) > 1
    for block in blocks:
        lines = block.split("\n")[2:]
        assert 1 <= len(lines) <= 2
        assert all(len(line) <= 42 for line in lines)


def test_vtt_readable_splits_and_keeps_plain_format() -> None:
    out = exporters.to_vtt(_long_words_result(), readable=True)
    assert out.startswith("WEBVTT\n\n")
    assert out.count(" --> ") > 1
    assert "," not in re.findall(r"\d\d:\d\d:\d\d.\d{3} --> \S+", out)[0]


def test_srt_default_keeps_one_cue_per_segment() -> None:
    assert exporters.to_srt(_long_words_result()).count(" --> ") == 1


def test_export_all_uses_readable_cues_only_when_asked(tmp_path: Path) -> None:
    result = _long_words_result()
    on = exporters.export_all(result, META, tmp_path / "on", ["srt", "vtt", "txt"], readable_subtitles=True)
    off = exporters.export_all(result, META, tmp_path / "off", ["srt", "vtt", "txt"], readable_subtitles=False)
    assert on["srt"].read_text(encoding="utf-8").count(" --> ") > 1
    assert off["srt"].read_text(encoding="utf-8").count(" --> ") == 1
    assert off["vtt"].read_text(encoding="utf-8").count(" --> ") == 1
    assert on["txt"].read_text(encoding="utf-8") == off["txt"].read_text(encoding="utf-8")
