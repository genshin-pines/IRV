"""
GPU vs CPU 速度对比测试
用法: python benchmark.py test_images/car.jpg
"""
import sys
import time
import cv2

image_path = sys.argv[1] if len(sys.argv) > 1 else None
if not image_path:
    print("用法: python benchmark.py <图片路径>")
    sys.exit(1)

img = cv2.imread(image_path)
if img is None:
    print(f"无法读取: {image_path}")
    sys.exit(1)

print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")
print()

# ── 预热 + 测试 GPU ──
print("=" * 50)
print("加载 GPU 加速版 HyperLPR3...")
from gpu_patch import catcher

# 预热（首次推理有初始化开销）
print("预热中...")
_ = catcher(img)

# 正式测试
print("GPU 推理中（10次取平均）...")
start = time.perf_counter()
for _ in range(10):
    results = catcher(img)
gpu_time = (time.perf_counter() - start) / 10

print(f"\nGPU (DirectML) 平均耗时: {gpu_time*1000:.1f} ms/帧")
print(f"  ≈ {1/gpu_time:.1f} FPS")
if results:
    print(f"  检测到: {[r[0] for r in results]}")
print()

# ── 与 CPU 对比 ──
print("=" * 50)
print("加载 CPU 版 HyperLPR3...")
import onnxruntime as ort

# 恢复原始 InferenceSession
_original_init_2 = ort.InferenceSession.__init__

def _cpu_init(self, *args, **kwargs):
    return _original_init_2(self, *args, **kwargs)

# 创建纯 CPU 实例
import importlib
import hyperlpr3 as h3

# 还原 patch，用 CPU　only
def _force_cpu(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    return _original_init_2(self, path_or_bytes, sess_options, ['CPUExecutionProvider'], provider_options, **kwargs)

ort.InferenceSession.__init__ = _force_cpu

# 重新加载以应用 CPU-only
importlib.reload(h3)
catcher_cpu = h3.LicensePlateCatcher()

print("预热中...")
_ = catcher_cpu(img)

print("CPU 推理中（10次取平均）...")
start = time.perf_counter()
for _ in range(10):
    results_cpu = catcher_cpu(img)
cpu_time = (time.perf_counter() - start) / 10

print(f"\nCPU 平均耗时: {cpu_time*1000:.1f} ms/帧")
print(f"  ≈ {1/cpu_time:.1f} FPS")
print()

# ── 汇总 ──
print("=" * 50)
print("速度对比:")
print(f"  GPU: {gpu_time*1000:.1f} ms  ({1/gpu_time:.1f} FPS)")
print(f"  CPU: {cpu_time*1000:.1f} ms  ({1/cpu_time:.1f} FPS)")
speedup = cpu_time / gpu_time
print(f"  加速比: {speedup:.1f}x")
print("=" * 50)
