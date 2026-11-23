"""Make the NVIDIA runtime DLLs shipped in site-packages loadable by ctranslate2.

Without this, the model constructs fine but the first encode fails with
"Library cublas64_12.dll is not found". Must run before importing faster_whisper.
"""

import os
import site
import sys
import sysconfig
from pathlib import Path

_registered: list[Path] = []
_handles: list[object] = []


def _site_packages_dirs() -> list[Path]:
    found: list[Path] = []
    candidates = [sysconfig.get_paths().get("purelib", "")]
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass
    for raw in candidates:
        if raw:
            p = Path(raw)
            if p.is_dir() and p not in found:
                found.append(p)
    return found


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for sp in _site_packages_dirs():
        nvidia = sp / "nvidia"
        if nvidia.is_dir():
            dirs.extend(sorted(b for b in nvidia.glob("*/bin") if b.is_dir()))
        ct2 = sp / "ctranslate2"
        if ct2.is_dir():
            dirs.append(ct2)
    unique: list[Path] = []
    for d in dirs:
        if d not in unique:
            unique.append(d)
    return unique


def register_nvidia_dll_dirs() -> list[Path]:
    if sys.platform != "win32":
        return []
    if _registered:
        return list(_registered)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for d in _candidate_dirs():
        try:
            _handles.append(os.add_dll_directory(str(d)))
        except OSError:
            continue
        if str(d) not in path_entries:
            path_entries.insert(0, str(d))
        _registered.append(d)
    os.environ["PATH"] = os.pathsep.join(path_entries)
    return list(_registered)
