import os
import sys

import pytest

from whisper_studio import cuda_setup


@pytest.mark.skipif(sys.platform != "win32", reason="windows only")
def test_registers_dirs_and_is_idempotent() -> None:
    first = cuda_setup.register_nvidia_dll_dirs()
    second = cuda_setup.register_nvidia_dll_dirs()
    assert first == second
    assert any(p.name == "ctranslate2" for p in first)
    assert any(p.parts[-2:] == ("cublas", "bin") for p in first)
    path_entries = os.environ["PATH"].split(os.pathsep)
    for p in first:
        assert path_entries.count(str(p)) == 1


def test_noop_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_setup.sys, "platform", "linux")
    monkeypatch.setattr(cuda_setup, "_registered", [])
    assert cuda_setup.register_nvidia_dll_dirs() == []


@pytest.mark.skipif(sys.platform != "win32", reason="windows only")
def test_frozen_bundle_root_is_searched(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (tmp_path / "ctranslate2").mkdir()
    added: list[str] = []
    monkeypatch.setattr(cuda_setup.sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(cuda_setup, "_site_packages_dirs", lambda: [])
    monkeypatch.setattr(cuda_setup, "_registered", [])
    monkeypatch.setattr(cuda_setup, "_handles", [])
    monkeypatch.setattr(cuda_setup.os, "add_dll_directory", lambda p: added.append(p) or object())
    monkeypatch.setenv("PATH", "")
    result = cuda_setup.register_nvidia_dll_dirs()
    assert tmp_path / "nvidia" / "cublas" / "bin" in result
    assert tmp_path / "ctranslate2" in result
    assert str(tmp_path / "ctranslate2") in added
