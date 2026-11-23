import sys
import threading
import tkinter as tk
from tkinter import messagebox
from types import TracebackType

from whisper_studio import APP_NAME
from whisper_studio.logging_setup import get_logger

log = get_logger("crash")

_FRIENDLY = (
    "Ocurrió un error inesperado.\n\n{summary}\n\n"
    "Los detalles quedaron guardados en data/logs/app.log. Si el problema continúa, "
    "comparte ese archivo para poder revisarlo."
)


def _show_dialog(exc: BaseException) -> None:
    summary = f"{type(exc).__name__}: {exc}"
    try:
        root = tk._default_root  # type: ignore[attr-defined]
        owns_root = root is None
        if owns_root:
            root = tk.Tk()
            root.withdraw()
        messagebox.showerror(APP_NAME, _FRIENDLY.format(summary=summary[:400]), parent=root)
        if owns_root:
            root.destroy()
    except Exception:
        log.exception("Could not show the error dialog")


def install_exception_hooks() -> None:
    def on_main(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        _show_dialog(exc)

    def on_thread(args: threading.ExceptHookArgs) -> None:
        if args.exc_value is not None:
            log.critical(
                "Uncaught exception in thread %s", args.thread.name if args.thread else "?",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

    sys.excepthook = on_main
    threading.excepthook = on_thread
