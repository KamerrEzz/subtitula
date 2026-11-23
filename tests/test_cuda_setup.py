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
