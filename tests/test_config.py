import json
from pathlib import Path

from whisper_studio.config import AppConfig, default_model_for, load_config, save_config


def test_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "settings.json")
    assert cfg.language == "es"
    assert cfg.formats == ["txt", "srt", "vtt", "md", "json"]
    assert cfg.existing_policy == "skip"
    assert cfg.model_user_set is False


def test_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    cfg = AppConfig(model="small", model_user_set=True, formats=["txt"], keep_media=True)
    save_config(cfg, p)
    assert load_config(p) == cfg


def test_corrupt_file_backed_up(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text("{not json", encoding="utf-8")
    cfg = load_config(p)
    assert cfg == AppConfig()
    assert (tmp_path / "settings.json.bak").read_text(encoding="utf-8") == "{not json"


def test_non_object_json_is_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text("[1,2]", encoding="utf-8")
    assert load_config(p).language == "es"
    assert (tmp_path / "settings.json.bak").exists()


def test_invalid_values_replaced_unknown_ignored(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(
        json.dumps(
            {
                "model": "gigantic",
                "language": "es",
                "device": 5,
                "formats": ["txt", "pdf", "srt"],
                "keep_media": "yes",
                "download_mode": "download_only",
                "mystery": 1,
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    default = AppConfig()
    assert cfg.model == default.model
    assert cfg.device == "auto"
    assert cfg.formats == ["txt", "srt"]
    assert cfg.keep_media is False
    assert cfg.download_mode == "download_only"
    assert not hasattr(cfg, "mystery")


def test_empty_formats_fall_back_to_default(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"formats": []}), encoding="utf-8")
    assert load_config(p).formats == ["txt", "srt", "vtt", "md", "json"]


def test_default_model_for() -> None:
    assert default_model_for(True) == "medium"
    assert default_model_for(False) == "small"


def test_readable_subtitles_defaults_to_true_and_tolerates_old_files(tmp_path: Path) -> None:
    assert AppConfig().readable_subtitles is True
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"model": "tiny", "model_user_set": True}), encoding="utf-8")
    assert load_config(p).readable_subtitles is True


def test_readable_subtitles_roundtrip_and_invalid_value(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    save_config(AppConfig(readable_subtitles=False), p)
    assert load_config(p).readable_subtitles is False
    p.write_text(json.dumps({"readable_subtitles": "no"}), encoding="utf-8")
    assert load_config(p).readable_subtitles is True
