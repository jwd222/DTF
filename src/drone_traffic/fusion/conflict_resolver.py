from __future__ import annotations

from typing import Any

from drone_traffic.fusion.association import associate_tracks


class ConflictResolver:
    def __init__(self, policy: str = "merge"):
        self._policy = policy
        self._global_track_map: dict[tuple[str, int], int] = {}
        self._next_global_id = 1

    def resolve(
        self,
        synced_messages: dict[str, dict[str, Any]],
        homographies: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sources = list(synced_messages.keys())
        if len(sources) < 2:
            track_data = []
            for source_id, msg in synced_messages.items():
                for t in msg.get("tracks", []):
                    gid = self._get_or_create_global_id(source_id, t.get("id", 0))
                    track_data.append(self._make_global_track(gid, source_id, t))
            return {"global_tracks": track_data, "events": []}

        source_a, source_b = sources[0], sources[1]
        tracks_a = synced_messages[source_a].get("tracks", [])
        tracks_b = synced_messages[source_b].get("tracks", [])

        for t in tracks_a:
            t["bev_position"] = self._project_to_bev(t, source_a, homographies)
        for t in tracks_b:
            t["bev_position"] = self._project_to_bev(t, source_b, homographies)

        matches, unmatched_a, unmatched_b = associate_tracks(tracks_a, tracks_b)

        global_tracks = []
        events = []

        for a_idx, b_idx in matches:
            gid = self._get_or_create_global_id(source_a, tracks_a[a_idx].get("id", 0))
            self._global_track_map[(source_b, tracks_b[b_idx].get("id", 0))] = gid
            merged = self._merge_tracks(gid, source_a, tracks_a[a_idx], source_b, tracks_b[b_idx])
            global_tracks.append(merged)
            events.append({
                "type": "track_merge",
                "global_id": gid,
                "sources": [source_a, source_b],
            })

        for idx in unmatched_a:
            gid = self._get_or_create_global_id(source_a, tracks_a[idx].get("id", 0))
            global_tracks.append(self._make_global_track(gid, source_a, tracks_a[idx]))

        for idx in unmatched_b:
            gid = self._get_or_create_global_id(source_b, tracks_b[idx].get("id", 0))
            global_tracks.append(self._make_global_track(gid, source_b, tracks_b[idx]))

        return {"global_tracks": global_tracks, "events": events}

    def _get_or_create_global_id(self, source_id: str, local_id: int) -> int:
        key = (source_id, local_id)
        if key not in self._global_track_map:
            self._global_track_map[key] = self._next_global_id
            self._next_global_id += 1
        return self._global_track_map[key]

    @staticmethod
    def _project_to_bev(
        track: dict[str, Any], source_id: str, homographies: dict[str, Any] | None
    ) -> list[float]:
        if homographies is None or source_id not in homographies:
            bbox = track.get("bbox", {})
            cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
            cy = bbox.get("y2", 0)
            return [cx, cy]
        import numpy as np

        H = homographies[source_id]
        if isinstance(H, np.ndarray):
            bbox = track.get("bbox", {})
            cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
            cy = bbox.get("y2", 0)
            pt = np.array([cx, cy, 1.0])
            projected = H @ pt
            if projected[2] != 0:
                return [projected[0] / projected[2], projected[1] / projected[2]]
        return [0.0, 0.0]

    @staticmethod
    def _merge_tracks(
        gid: int, src_a: str, t_a: dict, src_b: str, t_b: dict
    ) -> dict[str, Any]:
        return {
            "global_id": gid,
            "confidence": max(t_a.get("confidence", 0), t_b.get("confidence", 0)),
            "class_id": t_a.get("class_id", 0),
            "sources": {src_a: t_a, src_b: t_b},
            "bev_position": t_a.get("bev_position", [0, 0]),
        }

    @staticmethod
    def _make_global_track(gid: int, source_id: str, t: dict) -> dict[str, Any]:
        return {
            "global_id": gid,
            "confidence": t.get("confidence", 0),
            "class_id": t.get("class_id", 0),
            "sources": {source_id: t},
            "bev_position": t.get("bev_position", [0, 0]),
        }

    def reset(self) -> None:
        self._global_track_map.clear()
        self._next_global_id = 1
