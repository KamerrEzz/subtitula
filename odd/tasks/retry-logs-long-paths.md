# Retry failed jobs, log access and long Windows paths

## Objective
Playlist downloads failed with `[Errno 2] unable to open for writing`: the nested playlist folder
(added in 0.2.3) pushed paths past Windows MAX_PATH (260). Users also had no way to retry failed
jobs or reach the log without digging into Settings.

## Tasks
- [x] T1 Keep output folders within a path budget (group names capped, video title trimmed) and shorten the kept media file name.
- [x] T2 "Reintentar fallidos": reset failed jobs to pending and resume processing.
- [x] T3 "Abrir registro" button in the footer that opens app.log.

## Constraints
- TDD strict (RED → GREEN), runner `uv run pytest`.
- Route: direct inline (small edits, context already loaded).

## Evidence
- Failing path from the user's log: 269 characters; same job now 150 (dir) / 236 (worst file).
- Jobs finished under the previous names are still skipped (legacy_job_output_dir).
- `uv run pytest`: 259 passed. UI smoke at 1100x720: all footer and queue buttons visible.
