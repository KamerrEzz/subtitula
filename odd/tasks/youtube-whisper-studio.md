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

- Commits: 0b74082 bootstrap, 178933a core, 63369c9 ui, c90b551 docs (identity: repo-local, user-provided).

## Distribution (requested 2026-11-23)

- [x] T5 Frozen-mode paths: per-user writable data dir and default outputs when running as an installed exe;
  bundled deno/DLL discovery, `--self-test` diagnostic (delegated writer, with T6)
  - Evidence: RED (9 failed / collection error) then `uv run pytest -q` 196 passed. Commit: 2e34160.
- [x] T6 PyInstaller onedir build + NSIS 3.12 installer (`dist/YouTubeWhisperStudio-Setup-<version>.exe`),
  frozen smoke: launch, CUDA tiny transcription, YouTube probe (delegated writer)
  - Evidence: frozen `--self-test` -> cuda/int8_float16, probe "Me at the zoo"; installer 551 MB, onedir 1.2 GB
    (cuBLAS 790 MB is incompressible); silent install/launch/uninstall verified, user data kept.
- [x] T7 GitHub repo `kamerrezz/whisper-studio` (public, user choice), push, tag v0.1.0, release with installer
  asset (inline, `gh`)

## Rebrand + readable subtitles (requested 2026-11-23, v0.2.0)

- [x] T8 Rebrand to "Subtitula" (user choice): app name, window, installer, shortcuts, data/output dirs with
  migration from the old `YouTubeWhisperStudio` data dir, installer upgrade over 0.1.0, local videos as a
  first-class source in UI copy (delegated writer)
- [x] T9 Readable subtitles: SRT/VTT cues re-segmented with word timestamps (42 chars/line, 2 lines, 1-7 s,
  min gap, natural break points), option on by default (delegated writer, TDD)
- [ ] T10 Rebuild installer 0.2.0 (done, delegated) + GitHub release v0.2.0 (pending, inline)

  - T8 evidence: RED (9 failed) then 201 passed; commit 02f496d. T9: RED (collection error; 6 failed) then 231
    passed; commit 31f156d. T10 build: Subtitula-Setup-0.2.0.exe 551 MB; frozen self-test cuda/int8_float16,
    probe "Me at the zoo"; real 0.1.0 -> 0.2.0 silent upgrade verified (old key/dir/shortcuts gone), data
    migrated, reinstall in place and silent uninstall (data kept) verified.

- T10 release: v0.2.0 published with `Subtitula-Setup-0.2.0.exe` (sha256 37a722f2...b630), main pushed.
  - Note: yt-dlp-ejs 0.8.0 failed one rotated YouTube player (b485e332) as a warning; current player solves fine.
    Already latest yt-dlp; keep it updated in future releases.

## Not verified

- Real mouse drag & drop (only registration), "Cancelar todo" / close-while-processing dialogs, amber fallback
  badge in the UI, real playlist URL, real video-mode download with FFmpeg merge.
- Installer GUI paths: uninstall "Yes, delete data" answer, desktop-shortcut checkbox, finish-page run box.

## Acceptance

See the 14 criteria in the original request; record which ones were verified here.

## Progress / evidence

- (pending)
