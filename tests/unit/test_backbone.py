import torch

from drone_traffic.models.backbone_base import DummyBackbone


def test_dummy_backbone_forward():
    backbone = DummyBackbone(in_channels=3, channels=[64, 128])
    x = torch.randn(1, 3, 64, 64)
    features = backbone(x)

    assert "c1" in features
    assert "c2" in features
    assert features["c1"].shape[1] == 64
    assert features["c2"].shape[1] == 128


def test_dummy_backbone_properties():
    backbone = DummyBackbone(channels=[32, 64, 128], strides=[2, 4, 8])
    assert backbone.output_channels == [32, 64, 128]
    assert backbone.output_strides == [2, 4, 8]


def test_dummy_backbone_batch():
    backbone = DummyBackbone(channels=[64])
    x = torch.randn(4, 3, 32, 32)
    features = backbone(x)
    assert features["c1"].shape[0] == 4
