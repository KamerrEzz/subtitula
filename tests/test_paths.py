from pathlib import Path

import pytest

from whisper_studio import paths


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('a<b>c:d"e/f\\g|h?i*j', "abcdefghij"),
        ("  hola   mundo  ", "hola mundo"),
        ("video...", "video"),
        ("name. . ", "name"),
        ("CON", "_CON"),
        ("con", "_con"),
        ("LPT3", "_LPT3"),
        ("com1.txt", "_com1.txt"),
        ("", "sin-titulo"),
        ("???", "sin-titulo"),
        ("a\x00b\x1fc", "abc"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert paths.sanitize_filename(raw) == expected


def test_sanitize_truncates_and_strips_trailing() -> None:
    out = paths.sanitize_filename("x" * 50 + " " + "y" * 100, max_len=51)
    assert len(out) <= 51
    assert not out.endswith(" ")


def test_job_output_dir(tmp_path: Path) -> None:
    d = paths.job_output_dir(tmp_path, "Mi: video?", "abc123")
    assert d == tmp_path / "Mi video-[abc123]"


def test_unique_path(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    assert paths.unique_path(p) == p
    p.write_text("x")
    assert paths.unique_path(p) == tmp_path / "a (2).txt"
    (tmp_path / "a (2).txt").write_text("x")
    assert paths.unique_path(p) == tmp_path / "a (3).txt"


def test_atomic_write_text(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "f.txt"
    paths.atomic_write_text(p, "héllo")
    assert p.read_text(encoding="utf-8") == "héllo"
    paths.atomic_write_text(p, "new")
    assert p.read_text(encoding="utf-8") == "new"
    assert [x.name for x in p.parent.iterdir()] == ["f.txt"]


def test_app_dirs() -> None:
    assert paths.data_dir() == paths.app_root() / "data"
    assert paths.default_output_dir() == paths.app_root() / "outputs"
    assert (paths.app_root() / "pyproject.toml").exists()
