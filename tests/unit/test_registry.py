from drone_traffic.core.registry import (
    build_component,
    get_registry,
    register_backbone,
    register_detector,
    register_tracker,
)


def test_register_and_build():
    @register_backbone("test_bb")
    class TestBackbone:
        def __init__(self, size=64):
            self.size = size

    bb = build_component("backbone", "test_bb", size=128)
    assert bb.size == 128


def test_unknown_component():
    import pytest

    with pytest.raises(ValueError, match="Unknown backbone"):
        build_component("backbone", "nonexistent")


def test_get_registry():
    reg = get_registry("backbone")
    assert isinstance(reg, dict)


def test_register_tracker():
    @register_tracker("test_tracker")
    class TestTracker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    t = build_component("tracker", "test_tracker", max_age=10)
    assert t.kwargs["max_age"] == 10
