import socket
import threading
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from yt_dlp.utils import DownloadError, ExtractorError

from whisper_studio import downloader
from whisper_studio.errors import CancelledError, DownloadFailed


@pytest.mark.parametrize(
    "text,ok",
    [
        ("https://www.youtube.com/watch?v=jNQXAC9IVRw", True),
        ("http://youtube.com/watch?v=abc", True),
        ("https://youtu.be/jNQXAC9IVRw", True),
        ("https://music.youtube.com/watch?v=abc", True),
        ("https://m.youtube.com/playlist?list=PL123", True),
        ("  https://youtu.be/abc  ", True),
        ("youtube.com/watch?v=abc", False),
        ("hola mundo", False),
        ("", False),
        ("ftp://youtube.com/x", False),
        ("https://evil.example.com/?u=youtube.com", False),
        ("https://notyoutube.com/watch?v=1", False),
        ("https://youtube.com.evil.org/watch?v=1", False),
    ],
)
def test_is_supported_url(text: str, ok: bool) -> None:
    assert downloader.is_supported_url(text) is ok


def opts(mode: str, quality: str = "best", ffmpeg: bool = True) -> dict[str, Any]:
    return downloader.build_ydl_options(mode, quality, Path("out"), ffmpeg, lambda d: None)


def test_audio_options_have_no_postprocessor() -> None:
    o = opts("audio_transcribe")
    assert o["format"] == "bestaudio[ext=m4a]/bestaudio/best"
    assert not o.get("postprocessors")
    assert "merge_output_format" not in o


def test_video_with_ffmpeg_merges() -> None:
    o = opts("video_transcribe", "720")
    assert o["format"] == "bestvideo[height<=720]+bestaudio/best[height<=720]"
    assert o["merge_output_format"] == "mp4"
    assert opts("video_transcribe", "best")["format"] == "bestvideo+bestaudio/best"


def test_video_without_ffmpeg_uses_single_file() -> None:
    o = opts("video_transcribe", "480", ffmpeg=False)
    assert o["format"] == "best[height<=480][acodec!=none][vcodec!=none]/best"
    assert "merge_output_format" not in o
    assert opts("video_transcribe", "best", ffmpeg=False)["format"] == "best[acodec!=none][vcodec!=none]/best"


def test_download_only_downloads_video() -> None:
    assert "bestvideo" in opts("download_only", "1080")["format"]


def test_common_options() -> None:
    hook = lambda d: None  # noqa: E731
    o = downloader.build_ydl_options("audio_transcribe", "best", Path("out"), True, hook)
    assert o["outtmpl"] == str(Path("out") / "%(title).80B [%(id)s].%(ext)s")
    assert o["windowsfilenames"] is True
    assert o["restrictfilenames"] is False
    assert o["noplaylist"] is True
    assert o["retries"] == 10 and o["fragment_retries"] == 10
    assert o["continuedl"] is True
    assert o["overwrites"] is False
    assert o["writethumbnail"] is False
    assert o["socket_timeout"] == 30
    assert o["progress_hooks"] == [hook]
    for forbidden in ("cookiefile", "cookiesfrombrowser", "username", "password", "allow_unplayable_formats"):
        assert forbidden not in o


def test_ydl_options_are_not_shared_between_calls() -> None:
    a = opts("audio_transcribe")
    a["format"] = "x"
    assert opts("audio_transcribe")["format"] != "x"


@pytest.mark.parametrize(
    "exc,fragment",
    [
        (DownloadError("ERROR: [youtube] abc: Private video. Sign in if you've been granted access"), "privado"),
        (DownloadError("ERROR: [youtube] abc: Video unavailable"), "no está disponible"),
        (DownloadError("ERROR: [youtube] abc: This video has been removed by the uploader"), "eliminado"),
        (DownloadError("ERROR: [youtube] abc: Sign in to confirm your age"), "requiere iniciar sesión"),
        (DownloadError("ERROR: [youtube] abc: The uploader has not made this video available in your country"), "país"),
        (DownloadError("ERROR: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>"), "conexión"),
        (urllib.error.URLError("boom"), "conexión"),
        (socket.timeout("timed out"), "conexión"),
        (ExtractorError("weird failure"), "weird failure"),
        (ValueError("???"), "???"),
    ],
)
def test_error_mapping(exc: Exception, fragment: str) -> None:
    mapped = downloader.map_ytdlp_error(exc)
    assert isinstance(mapped, DownloadFailed)
    assert fragment in mapped.user_message
    assert mapped.detail


def test_login_message_states_no_bypass() -> None:
    mapped = downloader.map_ytdlp_error(DownloadError("Sign in to confirm your age"))
    assert "no se omiten restricciones de acceso" in mapped.user_message


class FakeYDL:
    info: dict[str, Any] = {}
    processed: dict[str, Any] | None = None
    extract_error: Exception | None = None
    process_hook = None
    last_opts: dict[str, Any] = {}

    def __init__(self, options: dict[str, Any]) -> None:
        type(self).last_opts = options
        self.options = options

    def __enter__(self) -> "FakeYDL":
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def extract_info(self, url: str, download: bool = True, **kw: Any) -> dict[str, Any]:
        if self.extract_error:
            raise self.extract_error
        return self.info

    def process_ie_result(self, info: dict[str, Any], download: bool = True) -> dict[str, Any]:
        if type(self).process_hook:
            type(self).process_hook(self.options)
        return type(self).processed if type(self).processed is not None else info

    def prepare_filename(self, info: dict[str, Any]) -> str:
        return info["_filename"]


@pytest.fixture
def fake_ydl(monkeypatch: pytest.MonkeyPatch):
    FakeYDL.info = {}
    FakeYDL.processed = None
    FakeYDL.extract_error = None
    FakeYDL.process_hook = None
    monkeypatch.setattr(downloader, "_ydl_class", lambda: FakeYDL)
    return FakeYDL


def test_probe_single_video(fake_ydl) -> None:
    fake_ydl.info = {
        "id": "abc", "title": "T", "duration": 19, "thumbnail": "http://t/x.jpg",
        "webpage_url": "https://www.youtube.com/watch?v=abc",
    }
    r = downloader.probe("https://www.youtube.com/watch?v=abc")
    assert r.kind == "video"
    assert (r.id, r.title, r.duration, r.thumbnail) == ("abc", "T", 19.0, "http://t/x.jpg")
    assert r.entries == [] and r.playlist_hint is False
    assert fake_ydl.last_opts["extract_flat"] == "in_playlist"
    assert fake_ydl.last_opts["noplaylist"] is True


def test_probe_watch_url_with_list_is_single_with_hint(fake_ydl) -> None:
    fake_ydl.info = {"id": "abc", "title": "T", "webpage_url": "u"}
    r = downloader.probe("https://www.youtube.com/watch?v=abc&list=PL1")
    assert r.kind == "video" and r.playlist_hint is True
    assert fake_ydl.last_opts["noplaylist"] is True


def test_probe_pure_playlist_filters_unavailable(fake_ydl) -> None:
    fake_ydl.info = {
        "_type": "playlist", "id": "PL1", "title": "Lista",
        "entries": [
            {"id": "a", "title": "A", "url": "https://www.youtube.com/watch?v=a", "duration": 10},
            None,
            {"id": "b", "title": "[Private video]", "url": "https://www.youtube.com/watch?v=b"},
            {"id": "c", "title": "[Deleted video]"},
            {"id": "d", "title": "D"},
        ],
    }
    r = downloader.probe("https://www.youtube.com/playlist?list=PL1")
    assert r.kind == "playlist" and r.title == "Lista"
    assert r.unavailable_count == 3
    assert [e.id for e in r.entries] == ["a", "d"]
    assert r.entries[1].url == "https://www.youtube.com/watch?v=d"
    assert r.entries[0].duration == 10.0
    assert fake_ydl.last_opts["noplaylist"] is False


def test_probe_maps_errors(fake_ydl) -> None:
    fake_ydl.extract_error = DownloadError("ERROR: Private video")
    with pytest.raises(DownloadFailed) as exc:
        downloader.probe("https://youtu.be/abc")
    assert "privado" in exc.value.user_message


def test_progress_hook_real_and_estimated() -> None:
    seen: list[downloader.DownloadProgress] = []
    hook = downloader.make_progress_hook(seen.append, threading.Event())
    hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 200, "speed": 1_258_291, "eta": 75})
    hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes_estimate": 100})
    hook({"status": "downloading", "downloaded_bytes": 50})
    hook({"status": "finished", "downloaded_bytes": 200, "total_bytes": 200})
    assert seen[0].fraction == 0.25 and seen[0].is_estimate is False
    assert seen[0].speed_text == "1.2 MB/s"
    assert seen[0].eta_text == "1:15"
    assert seen[1].fraction == 0.5 and seen[1].is_estimate is True
    assert seen[2].fraction is None
    assert seen[3].fraction == 1.0


def test_progress_hook_raises_on_cancel() -> None:
    ev = threading.Event()
    hook = downloader.make_progress_hook(lambda p: None, ev)
    hook({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 2})
    ev.set()
    with pytest.raises(CancelledError):
        hook({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 2})


@pytest.mark.parametrize(
    "value,text",
    [(None, ""), (512, "512 B/s"), (2048, "2.0 KB/s"), (3 * 1024 * 1024, "3.0 MB/s")],
)
def test_format_speed(value: float | None, text: str) -> None:
    assert downloader.format_speed(value) == text


def test_download_returns_final_path(fake_ydl, tmp_path: Path) -> None:
    final = tmp_path / "T [abc].m4a"
    final.write_bytes(b"x")
    fake_ydl.info = {"id": "abc", "_filename": str(final), "filesize": 10}
    fake_ydl.processed = {"id": "abc", "requested_downloads": [{"filepath": str(final)}]}
    r = downloader.download(
        "https://youtu.be/abc", "audio_transcribe", "best", tmp_path, lambda p: None, threading.Event(),
        ffmpeg_available=True,
    )
    assert r.path == final
    assert r.info["id"] == "abc"


def test_download_falls_back_to_prepare_filename(fake_ydl, tmp_path: Path) -> None:
    final = tmp_path / "T [abc].m4a"
    final.write_bytes(b"x")
    fake_ydl.info = {"id": "abc", "_filename": str(final)}
    r = downloader.download(
        "https://youtu.be/abc", "audio_transcribe", "best", tmp_path, lambda p: None, threading.Event(),
        ffmpeg_available=False,
    )
    assert r.path == final


def test_download_insufficient_space(fake_ydl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ydl.info = {"id": "abc", "_filename": "x", "filesize_approx": 10**15}
    called = []
    fake_ydl.process_hook = lambda o: called.append(1)
    with pytest.raises(DownloadFailed) as exc:
        downloader.download(
            "https://youtu.be/abc", "audio_transcribe", "best", tmp_path, lambda p: None,
            threading.Event(), ffmpeg_available=True,
        )
    assert exc.value.user_message == "Espacio insuficiente en disco."
    assert called == []


def test_download_sums_requested_formats(fake_ydl, tmp_path: Path) -> None:
    fake_ydl.info = {
        "id": "abc", "_filename": "x",
        "requested_formats": [{"filesize": 10**14}, {"filesize_approx": 10**14}],
    }
    with pytest.raises(DownloadFailed):
        downloader.download(
            "https://youtu.be/abc", "video_transcribe", "best", tmp_path, lambda p: None,
            threading.Event(), ffmpeg_available=True,
        )


def test_download_cancel_removes_only_new_partials(fake_ydl, tmp_path: Path) -> None:
    keep_old = tmp_path / "old [zzz].m4a.part"
    keep_old.write_bytes(b"old")
    keep_final = tmp_path / "prev [yyy].m4a"
    keep_final.write_bytes(b"ok")
    fake_ydl.info = {"id": "abc", "_filename": "x"}
    cancel = threading.Event()

    def during(options: dict[str, Any]) -> None:
        (tmp_path / "T [abc].m4a.part").write_bytes(b"p")
        (tmp_path / "T [abc].m4a.ytdl").write_bytes(b"p")
        (tmp_path / "T [abc].f251.webm.part").write_bytes(b"p")
        cancel.set()
        options["progress_hooks"][0]({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 2})

    fake_ydl.process_hook = during
    with pytest.raises(CancelledError):
        downloader.download(
            "https://youtu.be/abc", "audio_transcribe", "best", tmp_path, lambda p: None, cancel,
            ffmpeg_available=True,
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([keep_old.name, keep_final.name])


def test_download_cancel_wrapped_by_ytdlp_is_rethrown(fake_ydl, tmp_path: Path) -> None:
    fake_ydl.info = {"id": "abc", "_filename": "x"}
    cancel = threading.Event()
    cancel.set()

    def during(options: dict[str, Any]) -> None:
        raise DownloadError("wrapped")

    fake_ydl.process_hook = during
    with pytest.raises(CancelledError):
        downloader.download(
            "https://youtu.be/abc", "audio_transcribe", "best", tmp_path, lambda p: None, cancel,
            ffmpeg_available=True,
        )


def test_download_error_mapped(fake_ydl, tmp_path: Path) -> None:
    fake_ydl.extract_error = DownloadError("ERROR: Video unavailable")
    with pytest.raises(DownloadFailed) as exc:
        downloader.download(
            "https://youtu.be/abc", "audio_transcribe", "best", tmp_path, lambda p: None,
            threading.Event(), ffmpeg_available=True,
        )
    assert "no está disponible" in exc.value.user_message
