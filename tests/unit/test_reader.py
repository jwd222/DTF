from drone_traffic.ingestion.reader import VideoReader


def test_video_reader_with_synthetic(sample_video):
    with VideoReader(source=sample_video, target_fps=30, resolution=(640, 640)) as reader:
        packets = reader.read_all()

    assert len(packets) > 0
    assert packets[0].frame_id == 0
    assert packets[0].image.shape == (3, 640, 640)


def test_video_reader_fps_skip(sample_video):
    with VideoReader(source=sample_video, target_fps=10, resolution=(640, 640)) as reader:
        packets_low = reader.read_all()

    with VideoReader(source=sample_video, target_fps=30, resolution=(640, 640)) as reader:
        packets_high = reader.read_all()

    assert len(packets_low) <= len(packets_high)


def test_video_reader_properties(sample_video):
    reader = VideoReader(source=sample_video)
    assert reader.total_frames == 0

    with VideoReader(source=sample_video) as reader:
        assert reader.total_frames > 0
        assert reader.source_fps > 0


def test_video_reader_not_opened():
    import pytest

    reader = VideoReader(source="/nonexistent.mp4")
    with pytest.raises(RuntimeError):
        list(reader.read_frames())
