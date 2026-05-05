from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    homography = mapped_column(JSONB, nullable=True)
    bev_origin = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    config_snapshot = mapped_column(JSONB, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text)


class GlobalTrack(Base):
    __tablename__ = "global_tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("sessions.id")
    )
    global_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_label: Mapped[str | None] = mapped_column(String(50))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    total_observations: Mapped[int] = mapped_column(Integer, default=0)
    avg_velocity: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (UniqueConstraint("session_id", "global_id"),)


class TrackObservation(Base):
    __tablename__ = "track_observations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    global_track_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("global_tracks.id")
    )
    camera_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cameras.id")
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    frame_id: Mapped[int | None] = mapped_column(BigInteger)
    bbox_px = mapped_column(JSONB, nullable=True)
    bbox_bev = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    velocity_bev = mapped_column(JSONB, nullable=True)
    source_track_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_obs_track", "global_track_id"),
        Index("idx_obs_time", "timestamp"),
        Index("idx_obs_camera", "camera_id"),
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("sessions.id")
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    details = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("idx_events_session_time", "session_id", "timestamp"),
    )
