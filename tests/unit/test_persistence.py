from drone_traffic.persistence.models import (
    Camera,
    Event,
    GlobalTrack,
    Session,
    TrackObservation,
)


def test_camera_model():
    c = Camera(name="drone_1", description="Test camera")
    assert c.name == "drone_1"
    assert c.description == "Test camera"


def test_session_model():
    from datetime import datetime, timezone
    s = Session(started_at=datetime.now(timezone.utc))
    assert s.started_at is not None
    assert s.ended_at is None


def test_global_track_model():
    from datetime import datetime, timezone
    t = GlobalTrack(
        global_id=1,
        first_seen=datetime.now(timezone.utc),
        last_seen=datetime.now(timezone.utc),
    )
    assert t.global_id == 1
    assert t.total_observations == 0


def test_track_observation_model():
    from datetime import datetime, timezone
    o = TrackObservation(
        timestamp=datetime.now(timezone.utc),
        bbox_px={"x1": 10, "y1": 20, "x2": 100, "y2": 200},
        confidence=0.95,
    )
    assert o.confidence == 0.95


def test_event_model():
    from datetime import datetime, timezone
    e = Event(
        timestamp=datetime.now(timezone.utc),
        event_type="track_enter",
        details={"track_id": 1},
    )
    assert e.event_type == "track_enter"
