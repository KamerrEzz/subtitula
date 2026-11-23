from collections.abc import Callable, Iterator
from typing import Any

import pytest

ctk = pytest.importorskip("customtkinter")

from whisper_studio.downloader import ProbeResult  # noqa: E402
from whisper_studio.errors import DownloadFailed  # noqa: E402
from whisper_studio.ui import download_card as dc  # noqa: E402

URL = "https://www.youtube.com/watch?v=abc"


class SyncBridge:
    def run_async(self, work: Callable[[], Any], on_done: Callable[[Any, BaseException | None], None]) -> None:
        try:
            on_done(work(), None)
        except BaseException as exc:  # noqa: BLE001
            on_done(None, exc)


# One shared root: customtkinter cannot create a second CTk root in the same process.
@pytest.fixture(scope="module")
def root() -> Iterator[Any]:
    try:
        window = ctk.CTk()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Tk is not available: {exc}")
    yield window
    window.destroy()


def pump(window: Any, ms: int = dc.DEBOUNCE_MS + 300) -> None:
    window.after(ms, window.quit)
    window.mainloop()


def make_card(window: Any) -> Any:
    card = dc.DownloadCard(window, SyncBridge(), lambda *a: True, lambda m, l: None, lambda k, v: None)
    card.pack()
    card.apply_config("audio_transcribe", "best")
    return card


def test_pasting_the_same_link_again_retries_after_a_failed_probe(root: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes: list[Exception | None] = [DownloadFailed("Sin conexión a internet.", "offline"), None]
    calls: list[str] = []

    def fake_probe(url: str) -> ProbeResult:
        calls.append(url)
        error = outcomes.pop(0)
        if error is not None:
            raise error
        return ProbeResult("video", "abc", "Video de prueba", 10.0, None, url)

    monkeypatch.setattr(dc, "probe", fake_probe)
    card = make_card(root)

    card.set_text(URL)
    pump(root)
    assert card.error_label.winfo_ismapped()

    card.set_text(URL)
    pump(root)
    assert len(calls) == 2
    assert card.preview.winfo_ismapped()
    assert card.title_label.cget("text") == "Video de prueba"


def test_repasting_after_adding_shows_the_preview_again(root: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dc, "probe", lambda url: ProbeResult("video", "abc", "Video de prueba", 10.0, None, url))
    card = make_card(root)

    card.set_text(URL)
    pump(root)
    card.submit()
    pump(root, 50)
    assert card.entry.get() == ""

    card.set_text(URL)
    pump(root)
    assert card.preview.winfo_ismapped()
