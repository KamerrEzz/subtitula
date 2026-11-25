import subprocess
from pathlib import Path

import pytest

from whisper_studio import media


def test_is_media_file() -> None:
    assert media.is_media_file(Path("a.MP4"))
    assert media.is_media_file(Path("a.opus"))
    assert not media.is_media_file(Path("a.txt"))
    assert not media.is_media_file(Path("noext"))


def test_discover_media(tmp_path: Path) -> None:
    (tmp_path / "b.mp3").write_bytes(b"x")
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.wav").write_bytes(b"x")
    assert [p.name for p in media.discover_media(tmp_path)] == ["a.mp4", "b.mp3", "d.wav"]
    assert [p.name for p in media.discover_media(tmp_path, recursive=False)] == ["a.mp4", "b.mp3"]
    assert media.discover_media(tmp_path / "missing") == []


def test_check_free_space(tmp_path: Path) -> None:
    assert media.check_free_space(tmp_path, 1)
    assert not media.check_free_space(tmp_path, 10**18)


def test_check_free_space_nonexistent_uses_ancestor(tmp_path: Path) -> None:
    assert media.check_free_space(tmp_path / "a" / "b", 1)


def test_fingerprint_stable_and_sensitive(tmp_path: Path) -> None:
    f = tmp_path / "a.mp3"
    f.write_bytes(b"abc")
    fp = media.file_fingerprint(f)
    assert len(fp) == 8
    assert fp == media.file_fingerprint(f)
    f.write_bytes(b"abcd")
    assert media.file_fingerprint(f) != fp


def test_probe_duration_parses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(media, "which_ffprobe", lambda: "ffprobe")
    seen: dict = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 0, stdout="12.5\n", stderr="")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    assert media.probe_duration(tmp_path / "a.mp3") == 12.5
    assert seen["timeout"] > 0


@pytest.mark.parametrize("stdout", ["N/A\n", "", "abc", "0", "-1"])
def test_probe_duration_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: str) -> None:
    monkeypatch.setattr(media, "which_ffprobe", lambda: "ffprobe")
    monkeypatch.setattr(
        media.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    )
    assert media.probe_duration(tmp_path / "a.mp3") is None


def test_probe_duration_missing_tool_and_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(media, "which_ffprobe", lambda: None)
    assert media.probe_duration(tmp_path / "a.mp3") is None
    monkeypatch.setattr(media, "which_ffprobe", lambda: "ffprobe")

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(media.subprocess, "run", boom)
    assert media.probe_duration(tmp_path / "a.mp3") is None
    monkeypatch.setattr(
        media.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="3", stderr="e")
    )
    assert media.probe_duration(tmp_path / "a.mp3") is None


def test_folder_parts_start_at_the_added_folder(tmp_path: Path) -> None:
    root = tmp_path / "cursos" / "git"
    assert media.folder_parts(root / "M0S1-a.mp4", root) == ("git",)
    assert media.folder_parts(root / "modulo1" / "b.mp4", root) == ("git", "modulo1")
