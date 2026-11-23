# PyInstaller spec: onedir, windowed. Run from the repo root (see packaging/build.ps1).
import importlib.util
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent

datas, binaries, hiddenimports = [], [], []

for package in (
    "customtkinter",
    "tkinterdnd2",
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "yt_dlp",
    "yt_dlp_ejs",
    "certifi",
    "av",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# ctranslate2 only needs cuBLAS (+cuBLASLt); nvblas and nvrtc are never loaded, so they are left out.
cublas_bin = Path(importlib.util.find_spec("nvidia.cublas").submodule_search_locations[0]) / "bin"
for dll in ("cublas64_12.dll", "cublasLt64_12.dll"):
    binaries.append((str(cublas_bin / dll), "nvidia/cublas/bin"))

# yt-dlp runs deno as a subprocess to solve YouTube JS challenges, so it ships as data, not as a binary.
datas.append((str(ROOT / ".venv" / "Scripts" / "deno.exe"), "bin"))

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "_pytest", "IPython", "tkinter.test", "test", "unittest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Subtitula",
    console=False,
    icon=str(ROOT / "packaging" / "app.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Subtitula", upx=False)
