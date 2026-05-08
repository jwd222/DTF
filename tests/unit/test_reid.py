from unittest.mock import MagicMock, patch

import numpy as np

from drone_traffic.core.types import BBox, Detection


def test_reid_extractor_no_weights():
    with patch("drone_traffic.tracking.reid.ReIDExtractor._load_model"):
        from drone_traffic.tracking.reid import ReIDExtractor

        extractor = ReIDExtractor.__new__(ReIDExtractor)
        extractor._session = None
        extractor._embedding_dim = 512
        extractor._input_size = (256, 128)

        assert not extractor.is_ready

        det = Detection(
            bbox=BBox(x1=10, y1=10, x2=100, y2=200),
            confidence=0.9,
            class_id=0,
        )
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        features = extractor.extract_features([det], frame)
        assert features == [None]


def test_reid_extractor_with_session():
    with patch("drone_traffic.tracking.reid.ReIDExtractor._load_model"):
        from drone_traffic.tracking.reid import ReIDExtractor

        extractor = ReIDExtractor.__new__(ReIDExtractor)
        extractor._embedding_dim = 512
        extractor._input_size = (256, 128)

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input", shape=[1, 3, 256, 128])]
        mock_session.run.return_value = [np.random.randn(1, 512).astype(np.float32)]
        extractor._session = mock_session

        assert extractor.is_ready

        det = Detection(
            bbox=BBox(x1=10, y1=10, x2=100, y2=200),
            confidence=0.9,
            class_id=0,
        )
        frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        features = extractor.extract_features([det], frame)

        assert features[0] is not None
        assert features[0].shape == (512,)
        norm = np.linalg.norm(features[0])
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)


def test_reid_extractor_invalid_bbox():
    with patch("drone_traffic.tracking.reid.ReIDExtractor._load_model"):
        from drone_traffic.tracking.reid import ReIDExtractor

        extractor = ReIDExtractor.__new__(ReIDExtractor)
        extractor._embedding_dim = 512
        extractor._input_size = (256, 128)

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input", shape=[1, 3, 256, 128])]
        extractor._session = mock_session

        det = Detection(
            bbox=BBox(x1=200, y1=200, x2=100, y2=100),
            confidence=0.9,
            class_id=0,
        )
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        features = extractor.extract_features([det], frame)
        assert features[0] is None


def test_reid_extractor_empty_detections():
    with patch("drone_traffic.tracking.reid.ReIDExtractor._load_model"):
        from drone_traffic.tracking.reid import ReIDExtractor

        extractor = ReIDExtractor.__new__(ReIDExtractor)
        extractor._session = MagicMock()

        features = extractor.extract_features([], np.zeros((640, 480, 3), dtype=np.uint8))
        assert features == []
