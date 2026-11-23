import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from whisper_studio.paths import data_dir

_ROOT = "whisper_studio"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> Path:
    """Configure file and stderr logging once; returns the log file path."""
    directory = log_dir or data_dir() / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / "app.log"
    root = logging.getLogger(_ROOT)
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(_FORMAT)
    file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.propagate = False
    return log_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT}.{name}")
