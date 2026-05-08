from drone_traffic.core.config import AppConfig, load_config


def test_default_config():
    config = AppConfig()
    assert config.system.device == "cuda"
    assert config.system.fp16 is True
    assert config.input.resolution == [960, 960]
    assert config.models.detector.type == "yolo26_detector"
    assert config.tracking.type == "bot_sort"
    assert config.fusion.enabled is False


def test_config_from_yaml(tmp_path):
    yaml_content = """
system:
  device: "cpu"
  fp16: false
input:
  resolution: [1280, 1280]
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    config = load_config(str(yaml_file))
    assert config.system.device == "cpu"
    assert config.system.fp16 is False
    assert config.input.resolution == [1280, 1280]


def test_config_missing_file():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_config_streams():
    config = AppConfig(
        streams=[
            {"name": "drone_1", "source": "video1.mp4"},
            {"name": "drone_2", "source": "video2.mp4"},
        ]
    )
    assert len(config.streams) == 2
    assert config.streams[0].name == "drone_1"


def test_config_training_defaults():
    config = AppConfig()
    assert config.training.epochs == 300
    assert config.training.imgsz == 960
    assert config.training.optimizer == "MuSGD"


def test_config_tracking_reid():
    config = AppConfig()
    assert config.tracking.reid.enabled is False
    assert config.tracking.track_high_thresh == 0.5
    assert config.tracking.appearance_thresh == 0.25
