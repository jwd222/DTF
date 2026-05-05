from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from drone_traffic.persistence.database import get_session_factory
from drone_traffic.persistence.models import (
    Camera,
    Event,
    GlobalTrack,
    Session,
    TrackObservation,
)

router = APIRouter()


async def get_db() -> AsyncSession:
    factory = get_session_factory()
    async with factory() as session:
        yield session


@router.get("/health")
async def health() -> dict[str, Any]:
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": torch.cuda.get_device_properties(0).total_mem / (1024 ** 2),
            "vram_used_mb": torch.cuda.memory_allocated(0) / (1024 ** 2),
        }
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": gpu_info,
    }


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Session).order_by(Session.id.desc()).limit(limit))
    return result.scalars().all()


@router.get("/sessions/{session_id}/tracks")
async def get_session_tracks(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(GlobalTrack).where(GlobalTrack.session_id == session_id)
    )
    tracks = result.scalars().all()
    if not tracks:
        raise HTTPException(status_code=404, detail="Session not found or no tracks")
    return tracks


@router.get("/tracks/{track_id}/history")
async def get_track_history(
    track_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrackObservation)
        .where(TrackObservation.global_track_id == track_id)
        .order_by(TrackObservation.timestamp)
        .offset(offset)
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/tracks/{track_id}/trajectory")
async def get_track_trajectory(
    track_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrackObservation.timestamp, TrackObservation.bbox_bev)
        .where(TrackObservation.global_track_id == track_id)
        .order_by(TrackObservation.timestamp)
    )
    rows = result.all()
    trajectory = [{"timestamp": str(r[0]), "position": r[1]} for r in rows if r[1]]
    return {"track_id": track_id, "trajectory": trajectory}


@router.get("/stats")
async def get_stats(
    session_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(
        func.count(GlobalTrack.id).label("total_tracks"),
        func.avg(GlobalTrack.avg_velocity).label("avg_velocity"),
        func.sum(GlobalTrack.total_observations).label("total_observations"),
    )
    if session_id:
        query = query.where(GlobalTrack.session_id == session_id)

    result = await db.execute(query)
    row = result.one()
    return {
        "total_tracks": row.total_tracks or 0,
        "avg_velocity": float(row.avg_velocity) if row.avg_velocity else 0.0,
        "total_observations": row.total_observations or 0,
    }


@router.get("/events")
async def get_events(
    session_id: int | None = None,
    event_type: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    query = select(Event)
    if session_id:
        query = query.where(Event.session_id == session_id)
    if event_type:
        query = query.where(Event.event_type == event_type)
    if start_time:
        query = query.where(Event.timestamp >= start_time)
    if end_time:
        query = query.where(Event.timestamp <= end_time)

    query = query.order_by(Event.timestamp.desc()).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()
