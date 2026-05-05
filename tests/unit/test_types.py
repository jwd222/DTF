import numpy as np
from drone_traffic.core.types import BBox


def test_bbox_properties():
    bbox = BBox(x1=10, y1=20, x2=110, y2=70)
    assert bbox.width == 100
    assert bbox.height == 50
    assert bbox.cx == 60
    assert bbox.cy == 45
    assert bbox.area == 5000


def test_bbox_iou():
    a = BBox(x1=0, y1=0, x2=100, y2=100)
    b = BBox(x1=50, y1=50, x2=150, y2=150)
    iou = a.iou(b)
    expected = 50 * 50 / (100 * 100 + 100 * 100 - 50 * 50)
    assert abs(iou - expected) < 1e-6


def test_bbox_no_overlap():
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=20, y1=20, x2=30, y2=30)
    assert a.iou(b) == 0.0


def test_bbox_to_cxcyah():
    bbox = BBox(x1=0, y1=0, x2=100, y2=50)
    cx, cy, a, h = bbox.to_cxcyah()
    assert cx == 50
    assert cy == 25
    assert a == 2.0
    assert h == 50


def test_bbox_to_xywh():
    bbox = BBox(x1=10, y1=20, x2=110, y2=70)
    x, y, w, h = bbox.to_xywh()
    assert x == 10
    assert y == 20
    assert w == 100
    assert h == 50
