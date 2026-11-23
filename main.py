import ctypes
import os
import sys

from whisper_studio.cuda_setup import register_nvidia_dll_dirs
from whisper_studio.paths import bundled_bin_dir

# The DLL directories must be registered before anything imports faster_whisper or ctranslate2.
register_nvidia_dll_dirs()

# yt-dlp looks up its JavaScript runtime (deno, installed by uv) on PATH; launching the venv's
# python directly instead of through `uv run` would otherwise break YouTube challenge solving.
_scripts_dir = os.path.join(sys.prefix, "Scripts")
_bundled_bin = bundled_bin_dir()
_tool_dir = str(_bundled_bin) if _bundled_bin else (_scripts_dir if os.path.isdir(_scripts_dir) else None)
if _tool_dir:
    os.environ["PATH"] = _tool_dir + os.pathsep + os.environ.get("PATH", "")

from whisper_studio.logging_setup import get_logger, setup_logging  # noqa: E402
from whisper_studio.ui.crash import install_exception_hooks  # noqa: E402

APP_USER_MODEL_ID = "YouTubeWhisperStudio.App"


def _prepare_streams() -> None:
    # pythonw has no console: stdout/stderr are None and any print or stream handler would fail.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _set_taskbar_identity() -> None:
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def main() -> None:
    _prepare_streams()
    setup_logging()
    from whisper_studio import self_test

    request = self_test.parse_args(sys.argv[1:])
    if request is not None:
        sys.exit(self_test.run_self_test(*request))
    install_exception_hooks()
    _set_taskbar_identity()
    get_logger("main").info("Starting application")
    from whisper_studio.ui.app import App

    App().mainloop()


if __name__ == "__main__":
    main()
