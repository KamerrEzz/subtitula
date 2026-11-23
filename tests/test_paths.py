from pathlib import Path

import pytest

from whisper_studio import paths


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('a<b>c:d"e/f\\g|h?i*j', "abcdefghij"),
        ("  hola   mundo  ", "hola mundo"),
        ("video...", "video"),
        ("name. . ", "name"),
        ("CON", "_CON"),
        ("con", "_con"),
        ("LPT3", "_LPT3"),
        ("com1.txt", "_com1.txt"),
        ("", "sin-titulo"),
        ("???", "sin-titulo"),
        ("a\x00b\x1fc", "abc"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert paths.sanitize_filename(raw) == expected


def test_sanitize_truncates_and_strips_trailing() -> None:
    out = paths.sanitize_filename("x" * 50 + " " + "y" * 100, max_len=51)
    assert len(out) <= 51
    assert not out.endswith(" ")


def test_job_output_dir(tmp_path: Path) -> None:
    d = paths.job_output_dir(tmp_path, "Mi: video?", "abc123")
    assert d == tmp_path / "Mi video-[abc123]"


def test_unique_path(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    assert paths.unique_path(p) == p
    p.write_text("x")
    assert paths.unique_path(p) == tmp_path / "a (2).txt"
    (tmp_path / "a (2).txt").write_text("x")
    assert paths.unique_path(p) == tmp_path / "a (3).txt"


def test_atomic_write_text(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "f.txt"
    paths.atomic_write_text(p, "héllo")
    assert p.read_text(encoding="utf-8") == "héllo"
    paths.atomic_write_text(p, "new")
    assert p.read_text(encoding="utf-8") == "new"
    assert [x.name for x in p.parent.iterdir()] == ["f.txt"]


def test_app_dirs() -> None:
    assert paths.data_dir() == paths.app_root() / "data"
    assert paths.default_output_dir() == paths.app_root() / "outputs"
    assert (paths.app_root() / "pyproject.toml").exists()


def test_dev_dirs_stay_inside_the_repo() -> None:
    assert paths.data_dir(frozen=False) == paths.app_root() / "data"
    assert paths.default_output_dir(frozen=False) == paths.app_root() / "outputs"


def test_frozen_data_dir_uses_localappdata(tmp_path: Path) -> None:
    env = {"LOCALAPPDATA": str(tmp_path / "Local")}
    assert paths.data_dir(frozen=True, env=env) == tmp_path / "Local" / "Subtitula"


def test_frozen_data_dir_falls_back_to_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path))
    assert paths.data_dir(frozen=True, env={}) == tmp_path / "AppData" / "Local" / "Subtitula"


def test_frozen_output_dir_is_under_user_videos(tmp_path: Path) -> None:
    env = {"USERPROFILE": str(tmp_path)}
    out = paths.default_output_dir(frozen=True, env=env)
    assert out == tmp_path / "Videos" / "Subtitula"
    assert not out.exists()


def test_frozen_output_dir_falls_back_to_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path))
    assert paths.default_output_dir(frozen=True, env={}) == tmp_path / "Videos" / "Subtitula"


def test_is_frozen_reads_sys_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(paths.sys, "frozen", raising=False)
    assert paths.is_frozen() is False
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    assert paths.is_frozen() is True


def test_bundled_bin_dir_only_when_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.bundled_bin_dir(frozen=False) is None
    assert paths.bundled_bin_dir(frozen=True) == tmp_path / "bin"


def test_bundled_bin_dir_missing_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.bundled_bin_dir(frozen=True) is None


def test_package_version_matches_pyproject() -> None:
    import tomllib

    from whisper_studio import __version__

    pyproject = tomllib.loads((paths.app_root() / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__


def _legacy_env(tmp_path: Path) -> dict[str, str]:
    return {"LOCALAPPDATA": str(tmp_path / "Local")}


def test_migration_moves_legacy_dir_with_its_content(tmp_path: Path) -> None:
    legacy = tmp_path / "Local" / "YouTubeWhisperStudio"
    (legacy / "models").mkdir(parents=True)
    (legacy / "models" / "tiny.bin").write_bytes(b"model")
    (legacy / "settings.json").write_text("{}", encoding="utf-8")

    outcome = paths.migrate_legacy_data_dir(frozen=True, env=_legacy_env(tmp_path))

    assert outcome == "migrated"
    assert not legacy.exists()
    assert (tmp_path / "Local" / "Subtitula" / "models" / "tiny.bin").read_bytes() == b"model"
    assert (tmp_path / "Local" / "Subtitula" / "settings.json").exists()


def test_migration_is_noop_when_new_dir_exists(tmp_path: Path) -> None:
    legacy = tmp_path / "Local" / "YouTubeWhisperStudio"
    legacy.mkdir(parents=True)
    (legacy / "keep.txt").write_text("x", encoding="utf-8")
    (tmp_path / "Local" / "Subtitula").mkdir()

    assert paths.migrate_legacy_data_dir(frozen=True, env=_legacy_env(tmp_path)) == "not_needed"
    assert (legacy / "keep.txt").exists()


def test_migration_is_noop_without_legacy_dir(tmp_path: Path) -> None:
    (tmp_path / "Local").mkdir()
    assert paths.migrate_legacy_data_dir(frozen=True, env=_legacy_env(tmp_path)) == "not_needed"
    assert not (tmp_path / "Local" / "Subtitula").exists()


def test_migration_is_skipped_when_not_frozen(tmp_path: Path) -> None:
    legacy = tmp_path / "Local" / "YouTubeWhisperStudio"
    legacy.mkdir(parents=True)
    assert paths.migrate_legacy_data_dir(frozen=False, env=_legacy_env(tmp_path)) == "not_needed"
    assert legacy.exists()


def test_migration_failure_never_deletes_old_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    legacy = tmp_path / "Local" / "YouTubeWhisperStudio"
    legacy.mkdir(parents=True)
    (legacy / "keep.txt").write_text("x", encoding="utf-8")

    def locked(self: Path, target: Path) -> Path:
        raise PermissionError("in use")

    monkeypatch.setattr(paths.Path, "rename", locked)

    assert paths.migrate_legacy_data_dir(frozen=True, env=_legacy_env(tmp_path)) == "failed"
    assert (legacy / "keep.txt").exists()
    assert not (tmp_path / "Local" / "Subtitula").exists()
