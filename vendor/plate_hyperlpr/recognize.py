"""
HyperLPR3 最小识别脚本
用法：
    python recognize.py                    # 交互式输入图片路径
    python recognize.py path/to/image.jpg  # 指定图片路径

返回值格式 (Plate.to_result):
    [plate_code, rec_confidence, plate_type, det_bound_box]
     车牌号       识别置信度      颜色类型(int)  检测框[x1,y1,x2,y2]
"""
import sys
import time
import cv2
from gpu_patch import catcher  # GPU 加速版 HyperLPR3

# 车牌颜色映射表 (来自 hyperlpr3.common.typedef)
PLATE_COLOR_MAP = {
    -1: "未知",
     0: "蓝牌",
     1: "黄牌(单层)",
     2: "白牌(单层)",
     3: "绿牌(新能源)",
     4: "黑牌(港澳)",
     5: "香港(单层)",
     6: "香港(双层)",
     7: "澳门(单层)",
     8: "澳门(双层)",
     9: "黄牌(双层)",
}


def recognize_plate(image_path: str):
    """识别图片中的车牌并打印结果"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

    # 执行车牌识别（GPU 加速）
    print("正在识别车牌 (GPU-DirectML)...")
    t0 = time.perf_counter()
    results = catcher(img)
    elapsed = (time.perf_counter() - t0) * 1000

    if not results:
        print(f"未检测到车牌 (耗时 {elapsed:.0f}ms)")
        return

    # 打印结构化结果
    print(f"\n检测到 {len(results)} 个车牌 (耗时 {elapsed:.0f}ms):\n")
    print(f"{'序号':<6}{'车牌号':<14}{'置信度':<10}{'颜色':<14}{'检测框位置'}")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        plate_code = result[0]
        confidence = result[1]
        plate_type = result[2]
        bbox       = result[3]
        color_name = PLATE_COLOR_MAP.get(plate_type, f"未知({plate_type})")
        bbox_str   = f"({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])})"
        print(f"{i:<6}{plate_code:<14}{confidence:<10.2%}{color_name:<14}{bbox_str}")

    # 在图片上绘制标注框并保存
    for result in results:
        plate_code = result[0]
        confidence = result[1]
        plate_type = result[2]
        bbox       = result[3]
        color_name = PLATE_COLOR_MAP.get(plate_type, "未知")
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{plate_code} {color_name} ({confidence:.0%})"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_path = image_path.rsplit(".", 1)[0] + "_annotated.jpg"
    cv2.imwrite(output_path, img)
    print(f"\n标注图片已保存: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("用法: python recognize.py <图片路径>")
        print("示例: python recognize.py test_images/car.jpg\n")
        image_path = input("请输入图片路径: ").strip().strip('"')

    recognize_plate(image_path)
