import torch

from drone_traffic.models.nms import non_max_suppression


def test_nms_removes_overlapping():
    preds = torch.tensor(
        [
            [100, 100, 50, 50, 0.9, 0.1],
            [102, 102, 50, 50, 0.8, 0.1],
            [300, 300, 50, 50, 0.7, 0.1],
        ]
    ).float()

    result = non_max_suppression(preds, conf_threshold=0.5, iou_threshold=0.45)
    assert result.shape[0] == 2


def test_nms_below_threshold():
    preds = torch.tensor(
        [
            [100, 100, 50, 50, 0.1, 0.9],
            [200, 200, 50, 50, 0.1, 0.8],
        ]
    ).float()

    result = non_max_suppression(preds, conf_threshold=0.5, iou_threshold=0.45)
    assert result.shape[0] == 0


def test_nms_batch():
    preds = torch.randn(2, 10, 10)
    result = non_max_suppression(preds, conf_threshold=0.0)
    assert result.shape[0] == 2
