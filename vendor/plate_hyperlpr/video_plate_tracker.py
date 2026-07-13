"""
Vehicle tracker with decoupled detection/recognition API.

Detection loop: update_regions() every 100ms to keep positions fresh.
Recognition loop: assign_plate() when OCR completes.
Display loop: active_tracks() to get current bboxes + plate codes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading


def box_iou(a, b) -> float:
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


def center_distance(a, b) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def box_diag(box) -> float:
    return ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5


def normalize_bbox(bbox) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in bbox)


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
    tail_a, tail_b = a[1:], b[1:]
    if abs(len(tail_a) - len(tail_b)) <= 1 and edit_distance(tail_a, tail_b) <= 1:
        return True
    return edit_distance(a, b) <= 2


@dataclass
class VehicleTrack:
    track_id: int = 0
    first_time: float = 0.0
    last_time: float = 0.0
    last_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    missed: int = 0
    plate_code: str | None = None
    plate_conf: float = 0.0
    plate_type: int = -1
    ocr_candidates: list[dict] = field(default_factory=list)
    last_ocr_at: float = -999.0
    hits: int = 1
    ocr_pending: bool = False


class VehiclePlateTracker:
    def __init__(self, *, iou_threshold: float = 0.12, max_missed: int = 15,
                 ocr_cooldown: float = 2.0, min_hits_for_ocr: int = 2):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.ocr_cooldown = ocr_cooldown
        self.min_hits_for_ocr = min_hits_for_ocr
        self.next_id = 1
        self.tracks: list[VehicleTrack] = []
        self._lock = threading.Lock()

    # ── 检测线程调用：更新车辆位置 ──

    def update_regions(
        self,
        regions: list,
        timestamp: float,
        *,
        min_hits_for_ocr: int | None = None,
        ocr_cooldown: float | None = None,
        stop_ocr_confidence: float = 0.98,
    ):
        """只更新位置，不涉及 OCR。返回 (新track, 是否需要OCR) 列表。"""
        with self._lock:
            min_hits_for_ocr = min_hits_for_ocr or self.min_hits_for_ocr
            ocr_cooldown = self.ocr_cooldown if ocr_cooldown is None else ocr_cooldown
            matched = set()
            ocr_candidates = []

            for region in regions:
                bbox = normalize_bbox(region.bbox)
                track = self._match_track(bbox, excluded_track_ids=matched)

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
                    track.hits += 1

                needs_better_plate = (
                    not track.plate_code
                    or track.plate_conf < stop_ocr_confidence
                )
                if (
                    needs_better_plate
                    and not track.ocr_pending
                    and track.hits >= min_hits_for_ocr
                    and timestamp - track.last_ocr_at >= ocr_cooldown
                ):
                    track.ocr_pending = True
                    ocr_candidates.append((track, bbox))

                matched.add(track.track_id)

            # 未匹配的 track missed+1
            for track in self.tracks:
                if track.track_id not in matched:
                    track.missed += 1

            self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
            return ocr_candidates

    # ── 识别线程调用：绑定车牌号 ──

    def assign_plate(
        self,
        track_id: int,
        plate_code: str,
        confidence: float,
        plate_type: int = -1,
        timestamp: float = 0,
        *,
        task_bbox=None,
        task_timestamp: float | None = None,
        max_task_age: float | None = None,
        min_bind_iou: float = 0.05,
    ):
        """OCR 结果绑定到 track。"""
        with self._lock:
            for track in self.tracks:
                if track.track_id == track_id:
                    track.last_ocr_at = timestamp
                    track.ocr_pending = False
                    if max_task_age is not None and task_timestamp is not None:
                        if timestamp - task_timestamp > max_task_age:
                            return
                    if task_bbox is not None:
                        task_bbox = normalize_bbox(task_bbox)
                        dist = center_distance(track.last_bbox, task_bbox)
                        limit = max(box_diag(track.last_bbox), box_diag(task_bbox)) * 0.9
                        if box_iou(track.last_bbox, task_bbox) < min_bind_iou and dist > limit:
                            return
                    if not plate_code:
                        return
                    track.ocr_candidates.append({
                        "plate_code": plate_code,
                        "confidence": confidence,
                        "time_sec": timestamp,
                    })
                    # 多数投票
                    track.plate_code, track.plate_conf = self._vote(track.ocr_candidates)
                    track.plate_type = plate_type
                    return

    def cancel_ocr(self, track_id: int, timestamp: float = 0):
        """OCR 任务被丢弃或过期时释放该轨迹的 pending 状态。"""
        with self._lock:
            for track in self.tracks:
                if track.track_id == track_id:
                    track.last_ocr_at = timestamp
                    track.ocr_pending = False
                    return

    # ── 旧版视频识别兼容接口：server.py / recognize_video.py 仍在调用 ──

    def update(self, regions, plates: list[dict], timestamp: float):
        """兼容旧接口：同一帧内更新车辆区域，并把识别到的车牌候选绑定到轨迹。"""
        vehicle_regions = [region for region in regions if getattr(region, "source", "") == "vehicle"]
        with self._lock:
            region_to_track: dict[tuple[int, int, int, int], VehicleTrack] = {}
            matched_ids = set()

            for region in vehicle_regions:
                bbox = normalize_bbox(region.bbox)
                track = self._match_track(bbox, excluded_track_ids=matched_ids)
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
                    track.hits += 1

                region_to_track[bbox] = track
                matched_ids.add(track.track_id)

            for track in self.tracks:
                if track.track_id not in matched_ids:
                    track.missed += 1

            self.tracks = [
                track for track in self.tracks
                if track.missed <= self.max_missed or track.ocr_candidates
            ]

            for plate in plates:
                vehicle_bbox = plate.get("vehicle_bbox")
                if not vehicle_bbox:
                    continue
                bbox = normalize_bbox(vehicle_bbox)
                track = region_to_track.get(bbox) or self._match_track(bbox)
                if track is None:
                    continue

                item = dict(plate)
                item["time_sec"] = timestamp
                track.last_ocr_at = timestamp
                track.ocr_candidates.append(item)
                track.plate_code, track.plate_conf = self._vote(track.ocr_candidates)
                track.plate_type = item.get("plate_type", -1)

    def final_results(self) -> list[dict]:
        """兼容旧接口：按轨迹聚合候选，输出稳定车牌结果。"""
        with self._lock:
            results = []
            for track in self.tracks:
                if not track.ocr_candidates:
                    continue

                clusters: list[list[dict]] = []
                for candidate in track.ocr_candidates:
                    code = candidate.get("plate_code", "")
                    if not code:
                        continue
                    for cluster in clusters:
                        if similar_plate(code, cluster[0].get("plate_code", "")):
                            cluster.append(candidate)
                            break
                    else:
                        clusters.append([candidate])

                if not clusters:
                    continue

                best_cluster = max(
                    clusters,
                    key=lambda cluster: (
                        len(cluster),
                        sum(item.get("confidence", 0) for item in cluster) / len(cluster),
                        max(item.get("confidence", 0) for item in cluster),
                    ),
                )
                best_item = max(best_cluster, key=lambda item: item.get("confidence", 0))
                final_code, final_conf = self._vote(best_cluster)
                result = dict(best_item)
                result.update(
                    {
                        "plate_code": final_code,
                        "confidence": round(final_conf, 4),
                        "track_id": track.track_id,
                        "first_time": round(track.first_time, 2),
                        "last_time": round(track.last_time, 2),
                        "candidate_count": len(best_cluster),
                        "raw_candidates": sorted(
                            [
                                {
                                    "plate_code": item.get("plate_code", ""),
                                    "confidence": item.get("confidence", 0),
                                    "time_sec": item.get("time_sec"),
                                }
                                for item in best_cluster
                            ],
                            key=lambda item: item["time_sec"] if item["time_sec"] is not None else 0,
                        ),
                    }
                )
                results.append(result)

            return sorted(results, key=lambda item: item.get("first_time", 0))

    def _vote(self, candidates: list[dict]) -> tuple[str | None, float]:
        if not candidates:
            return None, 0.0
        best = max(candidates, key=lambda c: c["confidence"])
        target_len = len(best["plate_code"])

        same_len = [c for c in candidates if len(c["plate_code"]) == target_len]
        if not same_len:
            same_len = [best]

        chars = []
        for idx in range(target_len):
            weights: dict[str, float] = {}
            for item in same_len:
                code = item["plate_code"]
                if idx < len(code):
                    weights[code[idx]] = weights.get(code[idx], 0.0) + item["confidence"] ** 2
            chars.append(max(weights.items(), key=lambda p: p[1])[0] if weights else best["plate_code"][idx])

        final = "".join(chars)
        max_conf = max(c["confidence"] for c in same_len)
        return final, max_conf

    # ── 画面线程调用：获取当前活跃轨道 ──

    def active_tracks(self, timestamp: float | None = None, max_age: float | None = None):
        """返回 [(track_id, bbox, plate_code, plate_conf, plate_type), ...]"""
        with self._lock:
            results = []
            for t in self.tracks:
                if t.missed > self.max_missed:
                    continue
                if timestamp is not None and max_age is not None:
                    if timestamp - t.last_time > max_age:
                        continue
                results.append({
                    "track_id": t.track_id,
                    "bbox": list(t.last_bbox),
                    "plate_code": t.plate_code or "",
                    "plate_conf": t.plate_conf,
                    "plate_type": t.plate_type,
                })
            return sorted(results, key=lambda t: t["track_id"])

    # ── 内部 ──

    def _match_track(self, bbox, excluded_track_ids=None):
        excluded_track_ids = excluded_track_ids or set()
        best, best_score = None, -1.0
        for track in self.tracks:
            if track.track_id in excluded_track_ids:
                continue
            iou = box_iou(track.last_bbox, bbox)
            dist = center_distance(track.last_bbox, bbox)
            limit = max(box_diag(track.last_bbox), box_diag(bbox)) * 0.75
            if iou < self.iou_threshold and dist > limit:
                continue
            score = iou + max(0.0, 1.0 - dist / max(limit, 1.0)) * 0.25
            if score > best_score:
                best_score = score
                best = track
        return best
