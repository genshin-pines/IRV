"""
Image license plate recognition.

Usage:
    python recognize.py path/to/image.jpg

Pipeline:
    full image -> HyperLPR3
    vehicle detection -> vehicle crop -> upscale -> HyperLPR3
"""
import argparse
import time

import cv2

from gpu_patch import catcher
from vehicle_lpr import recognize_with_vehicle_crops


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


def draw_results(image, plates, vehicle_regions):
    annotated = image.copy()

    for region in vehicle_regions:
        if region.source != "vehicle":
            continue
        x1, y1, x2, y2 = region.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 160, 0), 1)

    for plate in plates:
        x1, y1, x2, y2 = plate["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{plate['plate_code']} {plate['confidence']:.0%} [{plate['source']}]"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return annotated


def recognize_plate(image_path: str, output_path: str | None = None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Can not read image: {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print("Running vehicle-first plate recognition...")

    started = time.perf_counter()
    plates, regions, rejected = recognize_with_vehicle_crops(
        image,
        catcher,
        return_rejected=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    vehicle_count = sum(1 for region in regions if region.source == "vehicle")
    print(f"Vehicles detected: {vehicle_count}")
    print(f"Rejected candidates: {len(rejected)}")

    if not plates:
        print(f"No plate detected ({elapsed_ms:.0f}ms)")
    else:
        print(f"\nDetected {len(plates)} plate(s) ({elapsed_ms:.0f}ms):\n")
        print(
            f"{'idx':<5}{'plate':<14}{'conf':<10}{'color':<15}"
            f"{'source':<10}{'bbox'}"
        )
        print("-" * 78)
        for idx, plate in enumerate(plates, 1):
            color = PLATE_COLOR_MAP.get(plate["plate_type"], "unknown")
            bbox = plate["bbox"]
            bbox_text = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
            print(
                f"{idx:<5}{plate['plate_code']:<14}"
                f"{plate['confidence']:<10.2%}{color:<15}"
                f"{plate['source']:<10}{bbox_text}"
            )

    if output_path is None:
        output_path = image_path.rsplit(".", 1)[0] + "_annotated.jpg"
    cv2.imwrite(output_path, draw_results(image, plates, regions))
    print(f"\nAnnotated image saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle-first plate recognition")
    parser.add_argument("image", help="image path")
    parser.add_argument("--output", "-o", default=None, help="annotated output path")
    args = parser.parse_args()

    recognize_plate(args.image, args.output)
