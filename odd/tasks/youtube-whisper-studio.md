# YouTube Whisper Studio

## Objective

Windows 11 desktop app (CustomTkinter) that downloads YouTube audio/video with yt-dlp and transcribes
locally with faster-whisper (CUDA when it really works, CPU otherwise). Spanish UI.

## Constraints

- Python 3.12, uv, `uv sync` + `uv run python main.py`, `start.bat`.
- No DRM/auth bypass. No silent model/precision change. No fake percentages.
- Small modular architecture: ui / download / transcription / export / queue / config.

## Verified environment facts

- GTX 1650 4 GB, driver 610.88, FFmpeg 8.1.1 on PATH.
- faster-whisper 1.2.1 + ctranslate2 4.9.1 (CUDA 12.4 build, bundles cuDNN 9).
- cuBLAS from `nvidia-cublas-cu12>=12.4,<12.9` (12.9 has no Windows wheel); DLL dir must be
  registered with `os.add_dll_directory`. Without it the model loads but the first encode fails.
- PyAV 19 breaks faster-whisper 1.2.1 -> `av>=12,<16`.
- `tiny` + `cuda` + `int8_float16` real inference verified.

## TDD

- Mode: strict (session config). Runner: `uv run pytest`.

## Delivery

- RDD: off (global). Strategy: single-pr (greenfield project, one feature branch `feat/whisper-studio`).

## Tasks

- [x] T1 Core: config, naming, media, cuda setup, engine, exporters, downloader, queue + tests (delegated writer: 2+ non-trivial files)
  - Evidence: `uv run pytest -q` 159 passed; engine `tiny/auto` -> cuda int8_float16; real JobQueue run wrote all
    outputs; real yt-dlp probe + audio download of jNQXAC9IVRw OK. Parent fix: video mode always keeps video
    (RED observed, then GREEN). Commit: pending (no git identity configured).
- [x] T2 UI: CustomTkinter app wired to the core, drag & drop, viewer, diagnostics (delegated writer)
  - Evidence: 21 helper tests (RED then GREEN); launch + processing smokes with screenshots; YouTube preview,
    3 jobs completed on CUDA, live transcript, dialogs rendered.
- [x] T3 Packaging: main.py, start.bat, README (es), .gitignore (inline, mechanical)
  - `yt-dlp[default,deno]` added so YouTube JS challenges are solved without external installs.
- [x] T4 Verification: full test run, app launch smoke, real local-file transcription on CUDA
  - `uv run pytest -q`: 182 passed. `medium` + cuda + int8_float16 real transcription OK (1.3 GB VRAM).
  - Found and fixed: a second CUDA encode after a library failure deadlocks inside ctranslate2; failure is now
    latched per process (OOM excluded). RED observed, then GREEN, real repro no longer hangs.

## Not verified

- Real mouse drag & drop (only registration), "Cancelar todo" / close-while-processing dialogs, amber fallback
  badge in the UI, real playlist URL, real video-mode download with FFmpeg merge.
- Commits: blocked, git user identity is not configured.

## Acceptance

See the 14 criteria in the original request; record which ones were verified here.

## Progress / evidence

- (pending)
