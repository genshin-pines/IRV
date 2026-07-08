from __future__ import annotations

from dataclasses import dataclass, field


def box_iou(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def box_diag(box: list[int] | tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def similar_plate(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False

    # Province character is the shakiest part in distant video. Compare the tail first.
    tail_a, tail_b = a[1:], b[1:]
    if abs(len(tail_a) - len(tail_b)) <= 1 and edit_distance(tail_a, tail_b) <= 1:
        return True

    return edit_distance(a, b) <= 2


def vote_plate(candidates: list[dict]) -> str:
    if not candidates:
        return ""
    best_candidate = max(candidates, key=lambda item: item["confidence"])
    target_len = len(best_candidate["plate_code"])
    same_len = [item for item in candidates if len(item["plate_code"]) == target_len]
    if not same_len:
        same_len = [best_candidate]

    chars = []
    for idx in range(target_len):
        weights: dict[str, float] = {}
        for item in same_len:
            code = item["plate_code"]
            if idx >= len(code):
                continue
            # Confidence is already filtered, square it to give very clear frames more pull.
            weights[code[idx]] = weights.get(code[idx], 0.0) + item["confidence"] ** 2
        if not weights:
            chars.append(best_candidate["plate_code"][idx])
            continue
        chars.append(max(weights.items(), key=lambda pair: pair[1])[0])
    return "".join(chars)


@dataclass
class VehicleTrack:
    track_id: int
    first_time: float
    last_time: float
    last_bbox: tuple[int, int, int, int]
    missed: int = 0
    candidates: list[dict] = field(default_factory=list)


class VehiclePlateTracker:
    def __init__(self, *, iou_threshold: float = 0.12, max_missed: int = 4):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: list[VehicleTrack] = []

    def _match_track(self, bbox: tuple[int, int, int, int]) -> VehicleTrack | None:
        best_track = None
        best_score = -1.0
        for track in self.tracks:
            iou = box_iou(track.last_bbox, bbox)
            dist = center_distance(track.last_bbox, bbox)
            limit = max(box_diag(track.last_bbox), box_diag(bbox)) * 0.75
            if iou < self.iou_threshold and dist > limit:
                continue
            score = iou + max(0.0, 1.0 - dist / max(limit, 1.0)) * 0.25
            if score > best_score:
                best_score = score
                best_track = track
        return best_track

    def update(self, regions, plates: list[dict], timestamp: float):
        vehicle_regions = [region for region in regions if region.source == "vehicle"]
        region_to_track: dict[tuple[int, int, int, int], VehicleTrack] = {}
        matched_ids = set()

        for region in vehicle_regions:
            bbox = tuple(region.bbox)
            track = self._match_track(bbox)
            if track is None:
                track = VehicleTrack(
                    track_id=self.next_id,
                    first_time=timestamp,
                    last_time=timestamp,
                    last_bbox=bbox,
                )
                self.next_id += 1
                self.tracks.append(track)
            else:
                track.last_time = timestamp
                track.last_bbox = bbox
                track.missed = 0

            region_to_track[bbox] = track
            matched_ids.add(track.track_id)

        for track in self.tracks:
            if track.track_id not in matched_ids:
                track.missed += 1

        self.tracks = [
            track for track in self.tracks
            if track.missed <= self.max_missed or track.candidates
        ]

        for plate in plates:
            vehicle_bbox = plate.get("vehicle_bbox")
            if not vehicle_bbox:
                continue
            bbox = tuple(vehicle_bbox)
            track = region_to_track.get(bbox)
            if track is None:
                track = self._match_track(bbox)
            if track is None:
                continue
            item = dict(plate)
            item["time_sec"] = timestamp
            track.candidates.append(item)

    def final_results(self) -> list[dict]:
        results = []
        for track in self.tracks:
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
            final_code = vote_plate(best_cluster)

            result = dict(best_item)
            result.update(
                {
                    "plate_code": final_code,
                    "confidence": round(max(item["confidence"] for item in best_cluster), 4),
                    "track_id": track.track_id,
                    "first_time": round(track.first_time, 2),
                    "last_time": round(track.last_time, 2),
                    "candidate_count": len(best_cluster),
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
                }
            )
            results.append(result)

        return sorted(results, key=lambda item: item["first_time"])
