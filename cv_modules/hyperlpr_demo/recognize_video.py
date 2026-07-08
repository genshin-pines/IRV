"""
Video license plate recognition.

Usage:
    python recognize_video.py path/to/video.mp4 --interval 0.5 --output result.json
"""
import argparse
import json
import time

import cv2

from gpu_patch import catcher
from vehicle_lpr import recognize_with_vehicle_crops
from video_plate_tracker import VehiclePlateTracker


PLATE_COLOR_MAP = {
    -1: "unknown",
    0: "blue",
    1: "yellow-single",
    2: "white-single",
    3: "green",
    4: "black",
    5: "hk-single",
    6: "hk-double",
    7: "macau-single",
    8: "macau-double",
    9: "yellow-double",
}


def recognize_video(video_path: str, interval: float = 0.5, output_json: str | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Can not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    step = max(1, int(fps * interval)) if fps > 0 else 1

    print(
        f"Video: {width}x{height}, {fps:.1f}fps, "
        f"{total_frames} frames, {duration:.1f}s"
    )
    print(f"Sampling interval: {interval}s, step={step} frame(s)")
    print("Running vehicle-first plate recognition...\n")

    started = time.perf_counter()
    tracker = VehiclePlateTracker()
    frame_idx = 0
    processed = 0
    vehicle_regions_total = 0
    rejected_total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            processed += 1
            timestamp = round(frame_idx / fps, 2) if fps > 0 else 0
            plates, regions, rejected = recognize_with_vehicle_crops(
                frame,
                catcher,
                return_rejected=True,
            )
            vehicle_regions_total += sum(1 for region in regions if region.source == "vehicle")
            rejected_total += len(rejected)

            for plate in plates:
                plate["plate_color"] = PLATE_COLOR_MAP.get(plate["plate_type"], "unknown")
                plate["frame"] = frame_idx
            tracker.update(regions, plates, timestamp)

            if processed % 10 == 0:
                print(
                    f"  processed={processed}, vehicles={vehicle_regions_total}, "
                    f"tracks={len(tracker.tracks)}"
                )

        frame_idx += 1

    cap.release()

    elapsed = time.perf_counter() - started
    plates = tracker.final_results()

    print("\n===== Done =====")
    print(f"Processed frames: {processed}")
    print(f"Vehicle regions: {vehicle_regions_total}")
    print(f"Rejected candidates: {rejected_total}")
    print(f"Unique plates: {len(plates)}")
    print(f"Elapsed: {elapsed:.1f}s\n")

    if plates:
        print(
            f"{'idx':<5}{'plate':<14}{'conf':<10}{'color':<15}"
            f"{'source':<10}{'track':<8}{'time'}"
        )
        print("-" * 70)
        for idx, plate in enumerate(plates, 1):
            print(
                f"{idx:<5}{plate['plate_code']:<14}"
                f"{plate['confidence']:<10.2%}{plate['plate_color']:<15}"
                f"{plate['source']:<10}{plate['track_id']:<8}"
                f"{plate['first_time']}s-{plate['last_time']}s"
            )

    if output_json:
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "video": video_path,
                    "fps": round(fps, 2),
                    "total_frames": total_frames,
                    "duration_sec": round(duration, 2),
                    "sample_interval_sec": interval,
                    "processed_frames": processed,
                    "vehicle_regions": vehicle_regions_total,
                    "rejected_count": rejected_total,
                    "unique_plates": len(plates),
                    "plates": plates,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nResult saved: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle-first video plate recognition")
    parser.add_argument("video", help="video path")
    parser.add_argument("--interval", "-i", type=float, default=0.5)
    parser.add_argument("--output", "-o", default=None, help="JSON output path")
    args = parser.parse_args()

    recognize_video(args.video, args.interval, args.output)
