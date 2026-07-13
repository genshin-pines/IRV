"""Track-id based plate aggregation for offline/video debugging."""
from __future__ import annotations

from dataclasses import dataclass, field

from vehicle_lpr import is_valid_plate_code
from video_plate_tracker import similar_plate


def vote_plate(candidates: list[dict]) -> tuple[str, float]:
    if not candidates:
        return "", 0.0

    best = max(candidates, key=lambda item: item.get("confidence", 0.0))
    target_len = len(best.get("plate_code", ""))
    same_len = [
        item for item in candidates
        if len(item.get("plate_code", "")) == target_len
    ] or [best]

    chars = []
    for idx in range(target_len):
        weights: dict[str, float] = {}
        for item in same_len:
            code = item.get("plate_code", "")
            if idx >= len(code):
                continue
            conf = float(item.get("confidence", 0.0))
            weights[code[idx]] = weights.get(code[idx], 0.0) + conf * conf
        chars.append(max(weights.items(), key=lambda pair: pair[1])[0] if weights else best["plate_code"][idx])

    return "".join(chars), max(float(item.get("confidence", 0.0)) for item in same_len)


@dataclass
class PlateTrack:
    track_id: int
    first_time: float
    last_time: float
    bbox: list[int]
    vehicle_confidence: float = 0.0
    hits: int = 0
    candidates: list[dict] = field(default_factory=list)


class PlateTrackStore:
    """Stores OCR candidates under externally supplied vehicle track ids."""

    def __init__(self, *, min_confidence: float = 0.98):
        self.min_confidence = min_confidence
        self.tracks: dict[int, PlateTrack] = {}

    def update_track(
        self,
        track_id: int,
        bbox,
        timestamp: float,
        vehicle_confidence: float = 0.0,
    ):
        bbox = [int(v) for v in bbox]
        track = self.tracks.get(track_id)
        if track is None:
            self.tracks[track_id] = PlateTrack(
                track_id=track_id,
                first_time=timestamp,
                last_time=timestamp,
                bbox=bbox,
                vehicle_confidence=vehicle_confidence,
                hits=1,
            )
            return

        track.last_time = timestamp
        track.bbox = bbox
        track.vehicle_confidence = vehicle_confidence
        track.hits += 1

    def add_plate(
        self,
        track_id: int,
        plate_code: str,
        confidence: float,
        plate_type: int,
        timestamp: float,
        *,
        plate_bbox=None,
        source: str = "vehicle_track",
    ):
        if not plate_code:
            return
        confidence = float(confidence)
        if confidence < self.min_confidence:
            return
        if not is_valid_plate_code(plate_code):
            return

        track = self.tracks.get(track_id)
        if track is None:
            return

        track.candidates.append({
            "plate_code": plate_code,
            "confidence": round(confidence, 4),
            "plate_type": int(plate_type),
            "bbox": [int(v) for v in plate_bbox] if plate_bbox else None,
            "source": source,
            "time_sec": timestamp,
        })

    def final_results(self) -> list[dict]:
        results = []
        for track in self.tracks.values():
            if not track.candidates:
                continue

            clusters: list[list[dict]] = []
            for candidate in track.candidates:
                for cluster in clusters:
                    if similar_plate(candidate["plate_code"], cluster[0]["plate_code"]):
                        cluster.append(candidate)
                        break
                else:
                    clusters.append([candidate])

            best_cluster = max(
                clusters,
                key=lambda cluster: (
                    len(cluster),
                    sum(item["confidence"] for item in cluster) / len(cluster),
                    max(item["confidence"] for item in cluster),
                ),
            )
            best_item = max(best_cluster, key=lambda item: item["confidence"])
            final_code, final_conf = vote_plate(best_cluster)
            result = dict(best_item)
            result.update({
                "plate_code": final_code,
                "confidence": round(final_conf, 4),
                "track_id": track.track_id,
                "first_time": round(track.first_time, 2),
                "last_time": round(track.last_time, 2),
                "candidate_count": len(best_cluster),
                "vehicle_bbox": track.bbox,
                "vehicle_confidence": round(track.vehicle_confidence, 4),
                "raw_candidates": sorted(
                    [
                        {
                            "plate_code": item["plate_code"],
                            "confidence": item["confidence"],
                            "time_sec": item.get("time_sec"),
                        }
                        for item in best_cluster
                    ],
                    key=lambda item: item["time_sec"] if item["time_sec"] is not None else 0,
                ),
            })
            results.append(result)

        return sorted(results, key=lambda item: item["first_time"])
