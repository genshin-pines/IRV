"""
HyperLPR3 GPU 加速补丁 (DirectML)
用法: 在导入 hyperlpr3 之前 import 此模块
     from gpu_patch import catcher  # 直接用带 GPU 的实例
"""
import onnxruntime as ort

# ── 验证 GPU 可用 ──
providers = ort.get_available_providers()
print(f"ONNX Runtime available providers: {providers}")

if 'DmlExecutionProvider' in providers:
    print("[OK] DirectML GPU acceleration ready (NVIDIA RTX 4060)")
else:
    print("[WARN] DmlExecutionProvider not available, falling back to CPU")

# ── Monkey-patch: 自动注入 DmlExecutionProvider ──
_original_init = ort.InferenceSession.__init__

def _patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    # 如果没有指定 provider 或只用了 CPU，注入 DirectML
    if providers is None:
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    elif providers == ['CPUExecutionProvider'] or 'CPUExecutionProvider' in providers:
        if 'DmlExecutionProvider' not in providers:
            providers = ['DmlExecutionProvider'] + list(providers)
    return _original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)

ort.InferenceSession.__init__ = _patched_init

# ── 导入 HyperLPR3 ──
import hyperlpr3 as lpr3

# ── Patch 1: 降低检测阈值，捕获远处小车牌 ──
from hyperlpr3.inference import multitask_detect
_original_post = multitask_detect.post_precessing

def _patched_post(dets, r, left, top, conf_thresh=0.1, iou_thresh=0.7):
    return _original_post(dets, r, left, top, conf_thresh, iou_thresh)

multitask_detect.post_precessing = _patched_post

# ── Patch 2: 降低 OCR 最小字符数门槛 ──
# 远处小车牌裁切分辨率低，OCR 可能读不全7位 → 放宽到5位
from hyperlpr3.inference import pipeline
_original_run = pipeline.LPRMultiTaskPipeline.run

def _patched_run(self, image):
    result = list()
    assert len(image.shape) == 3
    assert image is not None
    outputs = self.detector(image)
    for out in outputs:
        rect = out[:4].astype(int)
        score = out[4]
        land_marks = out[5:13].reshape(4, 2).astype(int)
        layer_num = int(out[13])
        pad = pipeline.get_rotate_crop_image(image, land_marks)
        if layer_num == pipeline.DOUBLE:
            h, w, _ = pad.shape
            line = int(h * 0.4)
            top = pad[:line, :, ]
            bottom = pad[line:, :]
            top_code, top_confidence = self.recognizer(top)
            bottom_code, bottom_confidence = self.recognizer(bottom)
            plate_code = top_code + bottom_code
            rec_confidence = (top_confidence + bottom_confidence) / 2
        else:
            plate_code, rec_confidence = self.recognizer(pad)
        if plate_code == '':
            continue
        # ★ 原: len(plate_code) >= 7  →  改: >= 4
        if len(plate_code) >= 4:
            plate_type = pipeline.code_filter(plate_code)
            if plate_type == pipeline.UNKNOWN:
                cls = self.classifier(pad)
                idx = int(pipeline.np.argmax(cls))
                if idx == pipeline.PLATE_TYPE_YELLOW:
                    plate_type = pipeline.YELLOW_DOUBLE if layer_num == pipeline.DOUBLE else pipeline.YELLOW_SINGLE
                elif idx == pipeline.PLATE_TYPE_BLUE:
                    plate_type = pipeline.BLUE
                elif idx == pipeline.PLATE_TYPE_GREEN:
                    plate_type = pipeline.GREEN
            plate = pipeline.Plate(vertex=land_marks, plate_code=plate_code,
                                   det_bound_box=pipeline.np.asarray(rect),
                                   rec_confidence=rec_confidence, dex_bound_confidence=score,
                                   plate_type=plate_type)
            result.append(plate.to_result())
    return result

pipeline.LPRMultiTaskPipeline.run = _patched_run

# ── 创建 GPU 加速的识别器 ──
catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
print("[OK] HyperLPR3 + GPU loaded (HIGH + len>=4 + relaxed thresholds)!\n")
