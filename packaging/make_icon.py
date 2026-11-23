"""Regenerates packaging/app.ico from the same drawing the app uses for its window icon."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from whisper_studio.ui.services import make_app_icon  # noqa: E402


def main() -> None:
    target = Path(__file__).resolve().parent / "app.ico"
    make_app_icon(256).save(target, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
