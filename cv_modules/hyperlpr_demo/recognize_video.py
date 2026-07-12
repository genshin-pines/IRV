"""
HyperLPR3 视频车牌识别
用法: python recognize_video.py <视频路径> [--interval 0.5] [--output results.json]
"""
import sys
import cv2
import json
import argparse
import hyperlpr3 as lpr3
from collections import OrderedDict

PLATE_COLOR_MAP = {
    -1: "未知", 0: "蓝牌", 1: "黄牌(单层)", 2: "白牌(单层)",
    3: "绿牌(新能源)", 4: "黑牌(港澳)", 5: "香港(单层)",
    6: "香港(双层)", 7: "澳门(单层)", 8: "澳门(双层)", 9: "黄牌(双层)",
}


def recognize_video(video_path: str, interval: float = 0.5, output_json: str = None):
    """识别视频中的车牌"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"视频信息: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"{fps:.1f}fps, {total_frames}帧, 时长 {duration:.1f}s")
    print(f"抽帧间隔: {interval}s (每 {int(fps * interval)} 帧抽1帧)")

    # 加载模型
    print("正在加载 HyperLPR3 模型...")
    catcher = lpr3.LicensePlateCatcher()
    print("模型加载完成，开始识别...\n")

    # 去重：同一车牌只保留最高置信度的一次
    plate_best = OrderedDict()  # plate_code -> {confidence, frame, time, bbox, color}
    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔抽帧
        if frame_idx % max(1, int(fps * interval)) == 0:
            processed_frames += 1
            results = catcher(frame)

            timestamp = frame_idx / fps  # 秒

            for r in results:
                plate_code = r[0]
                confidence = r[1]
                plate_type = int(r[2])
                bbox = [int(v) for v in r[3]]

                # 去重：只保留置信度最高的
                if plate_code not in plate_best or confidence > plate_best[plate_code]["confidence"]:
                    plate_best[plate_code] = {
                        "plate_code": plate_code,
                        "confidence": round(float(confidence), 4),
                        "plate_color": PLATE_COLOR_MAP.get(plate_type, "未知"),
                        "plate_type": plate_type,
                        "bbox": bbox,
                        "frame": frame_idx,
                        "time_sec": round(timestamp, 2),
                    }

            if processed_frames % 20 == 0:
                print(f"  已处理 {processed_frames} 帧, 累计发现 {len(plate_best)} 个不同车牌")

        frame_idx += 1

    cap.release()

    # 输出结果
    plates = list(plate_best.values())
    plates.sort(key=lambda x: x["time_sec"])

    print(f"\n===== 识别完成 =====")
    print(f"处理帧数: {processed_frames}, 发现 {len(plates)} 个不同车牌:\n")
    print(f"{'序号':<6}{'车牌号':<14}{'置信度':<10}{'颜色':<14}{'首现时间'}")
    print("-" * 60)
    for i, p in enumerate(plates, 1):
        print(f"{i:<6}{p['plate_code']:<14}{p['confidence']:<10.2%}{p['plate_color']:<14}{p['time_sec']}s")

    # 保存 JSON
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({
                "video": video_path,
                "fps": fps,
                "total_frames": total_frames,
                "duration_sec": round(duration, 1),
                "sample_interval_sec": interval,
                "processed_frames": processed_frames,
                "unique_plates": len(plates),
                "plates": plates,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频车牌识别")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--interval", "-i", type=float, default=0.5,
                        help="抽帧间隔(秒), 默认0.5s")
    parser.add_argument("--output", "-o", default=None,
                        help="JSON 输出路径, 如 results.json")
    args = parser.parse_args()

    recognize_video(args.video, args.interval, args.output)
