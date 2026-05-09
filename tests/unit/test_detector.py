from unittest.mock import MagicMock, patch

import numpy as np
import torch

from drone_traffic.core.types import BBox, Detection
from drone_traffic.models.yolo26_detector import VEHICLE_CLASSES


def test_vehicle_classes_defined():
    assert len(VEHICLE_CLASSES) == 6
    assert VEHICLE_CLASSES[0] == "car"
    assert VEHICLE_CLASSES[5] == "motorcycle"


def test_detector_forward_empty_results():
    with patch("drone_traffic.models.yolo26_detector.YOLOv26Detector._load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = []
        mock_load.return_value = mock_model

        from drone_traffic.models.yolo26_detector import YOLOv26Detector

        det = YOLOv26Detector.__new__(YOLOv26Detector)
        det._model = mock_model
        det._conf_threshold = 0.25
        det._iou_threshold = 0.45
        det._max_detections = 300
        det._num_classes = 6
        det._device = "cpu"

        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = det.forward(
            {},
            (640, 640, 3),
            1.0,
            (0, 0, 0, 0),
            raw_frame=frame,
        )
        assert results == []


def test_detector_forward_with_detections():
    with patch("drone_traffic.models.yolo26_detector.YOLOv26Detector._load_model"):
        from drone_traffic.models.yolo26_detector import YOLOv26Detector

        det = YOLOv26Detector.__new__(YOLOv26Detector)
        det._conf_threshold = 0.25
        det._iou_threshold = 0.45
        det._max_detections = 300
        det._num_classes = 6
        det._device = "cpu"

        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[100, 150, 200, 250], [300, 350, 400, 450]])
        mock_boxes.conf = torch.tensor([0.9, 0.6])
        mock_boxes.cls = torch.tensor([0, 2])
        mock_boxes.__len__ = MagicMock(return_value=2)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        det._model = MagicMock()
        det._model.predict.return_value = [mock_result]

        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = det.forward(
            {},
            (640, 640, 3),
            1.0,
            (0, 0, 0, 0),
            raw_frame=frame,
        )

        assert len(results) == 2
        assert results[0].class_id == 0
        assert results[0].class_label == "car"
        assert abs(results[0].confidence - 0.9) < 1e-6
        assert results[1].class_id == 2
        assert results[1].class_label == "truck"


def test_detector_confidence_filtering():
    with patch("drone_traffic.models.yolo26_detector.YOLOv26Detector._load_model"):
        from drone_traffic.models.yolo26_detector import YOLOv26Detector

        det = YOLOv26Detector.__new__(YOLOv26Detector)
        det._conf_threshold = 0.5
        det._iou_threshold = 0.45
        det._max_detections = 300
        det._num_classes = 6
        det._device = "cpu"

        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[100, 150, 200, 250]])
        mock_boxes.conf = torch.tensor([0.3])
        mock_boxes.cls = torch.tensor([0])
        mock_result.boxes = mock_boxes

        det._model = MagicMock()
        det._model.predict.return_value = [mock_result]

        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = det.forward(
            {},
            (640, 640, 3),
            1.0,
            (0, 0, 0, 0),
            raw_frame=frame,
        )

        assert len(results) == 0


def test_detector_no_raw_frame_no_tensor():
    with patch("drone_traffic.models.yolo26_detector.YOLOv26Detector._load_model"):
        from drone_traffic.models.yolo26_detector import YOLOv26Detector

        det = YOLOv26Detector.__new__(YOLOv26Detector)
        det._conf_threshold = 0.25
        det._iou_threshold = 0.45
        det._max_detections = 300
        det._num_classes = 6
        det._device = "cpu"
        det._model = MagicMock()

        results = det.forward(
            {},
            (640, 640, 3),
            1.0,
            (0, 0, 0, 0),
        )
        assert results == []


def test_detector_class_id_out_of_range():
    with patch("drone_traffic.models.yolo26_detector.YOLOv26Detector._load_model"):
        from drone_traffic.models.yolo26_detector import YOLOv26Detector

        det = YOLOv26Detector.__new__(YOLOv26Detector)
        det._conf_threshold = 0.25
        det._iou_threshold = 0.45
        det._max_detections = 300
        det._num_classes = 6
        det._device = "cpu"

        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[100, 150, 200, 250]])
        mock_boxes.conf = torch.tensor([0.9])
        mock_boxes.cls = torch.tensor([99])
        mock_result.boxes = mock_boxes

        det._model = MagicMock()
        det._model.predict.return_value = [mock_result]

        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = det.forward(
            {},
            (640, 640, 3),
            1.0,
            (0, 0, 0, 0),
            raw_frame=frame,
        )
        assert results == []
