"""
Vehicle-first license plate recognition helpers.

This module keeps the non-realtime image/video pipeline separate from the
RTSP server. It uses a COCO-pretrained YOLO model to find vehicles, crops and
upscales those regions, then runs HyperLPR3 on each crop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2


VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO
PROVINCES = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
SPECIAL_PLATE_PREFIXES = ("使", "领", "警", "学", "港", "澳")
_vehicle_model = None


@dataclass(frozen=True)
class Region:
    source: str
    bbox: tuple[int, int, int, int]
    vehicle_confidence: float | None = None


def get_vehicle_model(model_path: str = "yolov8n.pt"):
    """Lazily load the vehicle detector so importing this module stays cheap."""
    global _vehicle_model
    if _vehicle_model is None:
        from ultralytics import YOLO

        _vehicle_model = YOLO(model_path)
    return _vehicle_model


def expand_box(
    box: Iterable[float],
    width: int,
    height: int,
    margin_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin_ratio
    my = bh * margin_ratio
    return (
        max(0, int(x1 - mx)),
        max(0, int(y1 - my)),
        min(width, int(x2 + mx)),
        min(height, int(y2 + my)),
    )


def detect_vehicle_regions(
    image,
    *,
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    imgsz: int = 640,
    margin_ratio: float = 0.12,
    min_region_size: int = 24,
) -> list[Region]:
    """Return vehicle crop regions in original image coordinates."""
    height, width = image.shape[:2]
    model = get_vehicle_model(model_path)
    predictions = model.predict(
        image,
        classes=VEHICLE_CLASS_IDS,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )

    regions: list[Region] = []
    for prediction in predictions:
        if prediction.boxes is None:
            continue
        boxes = prediction.boxes.xyxy.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = expand_box(box, width, height, margin_ratio)
            if x2 - x1 < min_region_size or y2 - y1 < min_region_size:
                continue
            regions.append(
                Region(
                    source="vehicle",
                    bbox=(x1, y1, x2, y2),
                    vehicle_confidence=float(score),
                )
            )
    return regions


def map_plate_bbox_from_crop(
    plate_bbox: Iterable[float],
    region_bbox: tuple[int, int, int, int],
    scale: float,
) -> list[int]:
    rx1, ry1, _, _ = region_bbox
    px1, py1, px2, py2 = map(float, plate_bbox)
    return [
        int(rx1 + px1 / scale),
        int(ry1 + py1 / scale),
        int(rx1 + px2 / scale),
        int(ry1 + py2 / scale),
    ]


def dedupe_plate_results(results: list[dict]) -> list[dict]:
    """Keep the highest-confidence result for each plate code."""
    best: dict[str, dict] = {}
    for item in results:
        code = item["plate_code"]
        if code not in best or item["confidence"] > best[code]["confidence"]:
            best[code] = item
    return sorted(best.values(), key=lambda item: item["confidence"], reverse=True)


def is_valid_plate_code(code: str) -> bool:
    """Conservative Chinese plate format check for final display."""
    if not code:
        return False

    code = code.strip().upper()
    if len(code) not in (7, 8):
        return False

    if code[0] not in PROVINCES and not code.startswith(SPECIAL_PLATE_PREFIXES):
        return False

    if len(code) >= 2 and code[1] in PROVINCES:
        return False

    # Common civil plates: province + letter + 5/6 alnum chars.
    if code[0] in PROVINCES:
        if not ("A" <= code[1] <= "Z"):
            return False
        return all(ch.isdigit() or ("A" <= ch <= "Z") for ch in code[2:])

    return True


def filter_plate_results(
    results: list[dict],
    *,
    min_confidence: float = 0.90,
) -> tuple[list[dict], list[dict]]:
    """Split recognized plates into final results and rejected candidates."""
    accepted = []
    rejected = []
    for item in results:
        reasons = []
        if item["confidence"] < min_confidence:
            reasons.append("low_confidence")
        if not is_valid_plate_code(item["plate_code"]):
            reasons.append("invalid_format")

        if reasons:
            candidate = dict(item)
            candidate["reject_reasons"] = reasons
            rejected.append(candidate)
        else:
            accepted.append(item)
    return accepted, rejected


def recognize_with_vehicle_crops(
    image,
    plate_catcher,
    *,
    full_frame_mode: str = "fallback",
    vehicle_model_path: str = "yolov8n.pt",
    vehicle_conf: float = 0.25,
    vehicle_imgsz: int = 640,
    crop_scale: float = 2.0,
    min_plate_confidence: float = 0.90,
    return_rejected: bool = False,
) -> tuple[list[dict], list[Region]] | tuple[list[dict], list[Region], list[dict]]:
    """
    Recognize plates from vehicle crops, with optional full-frame fallback.

    Returned plate bbox values are always mapped to the original image.
    """
    if full_frame_mode not in {"fallback", "always", "never"}:
        raise ValueError("full_frame_mode must be one of: fallback, always, never")

    height, width = image.shape[:2]
    full_region = Region(source="full", bbox=(0, 0, width, height))
    vehicle_regions = detect_vehicle_regions(
        image,
        model_path=vehicle_model_path,
        conf=vehicle_conf,
        imgsz=vehicle_imgsz,
    )

    regions: list[Region] = []
    if full_frame_mode == "always":
        regions.append(full_region)
    regions.extend(vehicle_regions)

    def recognize_region(region: Region) -> list[dict]:
        x1, y1, x2, y2 = region.bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        scale = 1.0 if region.source == "full" else crop_scale
        if scale != 1.0:
            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )

        results = []
        for raw in plate_catcher(crop):
            plate_bbox = [int(v) for v in raw[3]]
            if region.source != "full":
                plate_bbox = map_plate_bbox_from_crop(plate_bbox, region.bbox, scale)

            results.append(
                {
                    "plate_code": raw[0],
                    "confidence": round(float(raw[1]), 4),
                    "plate_type": int(raw[2]),
                    "bbox": plate_bbox,
                    "source": region.source,
                    "vehicle_bbox": list(region.bbox)
                    if region.source == "vehicle"
                    else None,
                    "vehicle_confidence": round(region.vehicle_confidence, 4)
                    if region.vehicle_confidence is not None
                    else None,
                }
            )
        return results

    plate_results: list[dict] = []
    for region in regions:
        plate_results.extend(recognize_region(region))

    if full_frame_mode == "fallback" and not plate_results:
        regions.append(full_region)
        plate_results.extend(recognize_region(full_region))

    deduped = dedupe_plate_results(plate_results)
    accepted, rejected = filter_plate_results(
        deduped,
        min_confidence=min_plate_confidence,
    )

    if return_rejected:
        return accepted, regions, rejected
    return accepted, regions
