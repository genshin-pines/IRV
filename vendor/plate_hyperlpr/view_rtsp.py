"""
RTSP 摄像头画面查看器 (matplotlib 版)
用法: python view_rtsp.py          # 列出所有摄像头，选择一路观看
      python view_rtsp.py 1        # 直接看第1路
      python view_rtsp.py 桥面      # 模糊匹配名称
"""
import sys
import cv2
import matplotlib.pyplot as plt

CAMERAS = [
    ("1",  "桥面",       "rtsp://10.126.59.120:8554/live/live1"),
    ("2",  "停车场出口", "rtsp://10.126.59.120:8554/live/live2"),
    ("3",  "行人检测",   "rtsp://10.126.59.120:8554/live/live3"),
    ("4",  "消防车识别", "rtsp://10.126.59.120:8554/live/live4"),
    ("5",  "桥出口",     "rtsp://10.126.59.120:8554/live/live5"),
    ("6",  "桥入口",     "rtsp://10.126.59.120:8554/live/live6"),
    ("7",  "道路2",      "rtsp://10.126.59.120:8554/live/live7"),
    ("8",  "隧道(事故)",  "rtsp://10.126.59.120:8554/live/live8"),
    ("9",  "隧道(车载)",  "rtsp://10.126.59.120:8554/live/live9"),
    ("10", "道路1",       "rtsp://10.126.59.120:8554/live/live10"),
    ("11", "停车场入口",  "rtsp://10.126.59.120:8554/live/live11"),
    ("12", "道路1",       "rtsp://10.126.59.120:8554/live/live12"),
]


def select_camera(arg=None):
    """根据参数选择摄像头"""
    if arg is None:
        print("\n可用摄像头列表:\n")
        for idx, name, url in CAMERAS:
            print(f"  [{idx:>2}] {name:<10}  {url}")
        print()
        arg = input("输入编号或名称: ").strip()

    for idx, name, url in CAMERAS:
        if arg == idx or arg in name:
            return name, url

    print(f"未找到匹配的摄像头: {arg}")
    return None, None


def view_stream(url: str, name: str):
    """打开 RTSP 流并用 matplotlib 显示画面"""
    print(f"\n正在连接 {name}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"无法连接！请确认是否在沙盘内网(10.126.59.x 网段)")
        return

    print(f"已连接 [{name}]")
    print("  关闭窗口 = 退出 | 鼠标点击窗口后按 S = 截图保存\n")

    # matplotlib 交互模式
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title(f"沙盘摄像头 - {name}")

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        cap.release()
        plt.close()
        return

    # BGR → RGB（OpenCV 是 BGR，matplotlib 是 RGB）
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = ax.imshow(rgb)
    ax.set_title(f"{name}  |  关闭窗口退出  |  点击窗口后按 S 截图", fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    paused = False
    running = True

    def on_key(event):
        nonlocal paused
        if event.key == "s":
            filename = f"screenshot_{name}.jpg"
            cv2.imwrite(filename, frame)
            print(f"已保存: {filename}")
        elif event.key == " ":
            paused = not paused
            print("暂停" if paused else "继续")

    fig.canvas.mpl_connect("key_press_event", on_key)

    while running:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("视频流断开")
                break

        # 更新画面
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im.set_data(rgb)
        fig.canvas.draw_idle()

        try:
            plt.pause(0.03)  # 约 30fps
        except Exception:
            break

        # 检测窗口是否被关闭
        if not plt.fignum_exists(fig.number):
            running = False

    cap.release()
    plt.close("all")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    name, url = select_camera(arg)
    if url:
        view_stream(url, name)
