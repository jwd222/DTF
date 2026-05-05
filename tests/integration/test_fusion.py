from drone_traffic.fusion.temporal_sync import TemporalSyncBuffer


def test_sync_buffer_collects():
    buf = TemporalSyncBuffer(max_time_diff=0.1, sources_expected=2)
    buf.add("drone_1", {"drone_id": "drone_1", "timestamp": 1.0, "frame_id": 0, "tracks": []})
    buf.add("drone_2", {"drone_id": "drone_2", "timestamp": 1.02, "frame_id": 0, "tracks": []})
    result = buf.try_flush()
    assert result is not None
    assert "drone_1" in result
    assert "drone_2" in result


def test_sync_buffer_incomplete():
    buf = TemporalSyncBuffer(max_time_diff=0.1, sources_expected=2)
    buf.add("drone_1", {"drone_id": "drone_1", "timestamp": 1.0, "frame_id": 0, "tracks": []})
    result = buf.try_flush()
    assert result is None


def test_sync_buffer_reset():
    buf = TemporalSyncBuffer(max_time_diff=0.1, sources_expected=2)
    buf.add("drone_1", {"drone_id": "drone_1", "timestamp": 1.0, "frame_id": 0, "tracks": []})
    buf.reset()
    assert len(buf._latest) == 0
