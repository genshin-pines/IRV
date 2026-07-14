from alert_agent.rules import RuleEngine


def _ids(logs):
    """返回 RuleEngine.analyze 产生的所有 rule_id 集合"""
    return {result.rule_id for result in RuleEngine().analyze(logs)}


def _titles(logs):
    """返回 RuleEngine.analyze 产生的所有告警标题集合"""
    return {result.title for result in RuleEngine().analyze(logs)}


# ═══════════════════════════════════════════════════════════════
# PlatePipelineFailureRule — 车牌管线硬失败
# ═══════════════════════════════════════════════════════════════

def test_plate_pipeline_decode_failed():
    """图片解码失败 → 1 条 plate ERROR 即触发"""
    logs = [
        {"module": "plate", "level": "ERROR",
         "message": "plate recognition failed: image decode failed filename=corrupt.jpg"},
    ]
    assert "车牌识别管线异常" in _titles(logs)


def test_plate_pipeline_cannot_open():
    """视频文件打不开 → 1 条 plate ERROR 即触发"""
    logs = [
        {"module": "plate", "level": "ERROR",
         "message": "plate video failed: cannot open video filename=bad.mp4"},
    ]
    assert "车牌识别管线异常" in _titles(logs)


def test_plate_pipeline_load_failed():
    """HyperLPR3 模型加载失败 → 1 条 plate ERROR 即触发"""
    logs = [
        {"module": "plate", "level": "ERROR",
         "message": "车牌识别模块加载失败: No module named 'gpu_patch'"},
    ]
    assert "车牌识别管线异常" in _titles(logs)


def test_plate_pipeline_warmup_failed():
    """camera 模型预热失败 → 1 条 camera ERROR 即触发"""
    logs = [
        {"module": "camera", "level": "ERROR",
         "message": "model warmup failed, will retry on first stream: ONNX load error"},
    ]
    assert "车牌识别管线异常" in _titles(logs)


def test_plate_pipeline_not_triggered_by_info():
    """plate INFO 日志不触发"""
    logs = [
        {"module": "plate", "level": "INFO",
         "message": "plate image success: filename=ok.jpg count=1 rejected=0"},
    ]
    assert "车牌识别管线异常" not in _titles(logs)


# ═══════════════════════════════════════════════════════════════
# LLMDegradationRule — Token 超额 vs 通用降级
# ═══════════════════════════════════════════════════════════════

def test_llm_token_exceed():
    """LLM 返回 context length exceeded → Token 超额告警"""
    logs = [
        {"module": "llm", "level": "WARNING",
         "message": "LLM downgraded: 400 Client Error: maximum context length exceeded"},
    ]
    titles = _titles(logs)
    assert "LLM Token 超额或上下文超限" in titles


def test_llm_auth_fail():
    """鉴权失败 → 通用 LLM 异常告警（非 Token 超额）"""
    logs = [
        {"module": "llm", "level": "WARNING",
         "message": "LLM downgraded, status=401"},
    ]
    titles = _titles(logs)
    assert "LLM 接口异常或已降级" in titles


def test_llm_rate_limit():
    """速率限制 → 通用 LLM 异常告警"""
    logs = [
        {"module": "llm", "level": "ERROR",
         "message": "rate limit exceeded for model deepseek-chat"},
    ]
    assert "LLM 接口异常或已降级" in _titles(logs)


# ═══════════════════════════════════════════════════════════════
# TrafficPoliceAnomalyRule — 交警手势异常
# ═══════════════════════════════════════════════════════════════

def test_traffic_police_error():
    """模型加载失败 → 1 条 ERROR 即触发"""
    logs = [
        {"module": "traffic_police", "level": "ERROR",
         "message": "交警手势模型加载失败: torch unavailable"},
    ]
    assert "交警手势识别异常" in _titles(logs)


def test_traffic_police_stream_start_failed():
    """流启动失败 → 1 条 ERROR 即触发"""
    logs = [
        {"module": "traffic_police", "level": "ERROR",
         "message": "交警手势流启动失败: cannot open RTSP stream"},
    ]
    assert "交警手势识别异常" in _titles(logs)


def test_traffic_police_low_confidence():
    """≥2 条低置信度 WARNING → 触发"""
    logs = [
        {"module": "traffic_police", "level": "WARNING",
         "message": "traffic police gesture frame=30 id=0 label=无手势 confidence=0.45"}
        for _ in range(2)
    ]
    assert "traffic_police_anomaly" in _ids(logs)


def test_traffic_police_low_confidence_below_threshold():
    """1 条低置信度 → 不触发"""
    logs = [
        {"module": "traffic_police", "level": "WARNING",
         "message": "traffic police gesture frame=30 id=0 label=无手势 confidence=0.45"}
    ]
    assert "traffic_police_anomaly" not in _ids(logs)


# ═══════════════════════════════════════════════════════════════
# GestureLowConfidenceRule — 手势管线硬失败
# ═══════════════════════════════════════════════════════════════

def test_gesture_decode_failed():
    """手势帧解码失败 → 1 条 gesture ERROR 即触发"""
    logs = [
        {"module": "gesture", "level": "ERROR",
         "message": "gesture frame failed: image decode failed filename=bad.jpg"},
    ]
    assert "手势识别管线异常" in _titles(logs)


def test_gesture_low_confidence_still_works():
    """置信度持续低分支不受 decode failed 改动影响"""
    logs = [
        {"module": "gesture", "level": "WARNING",
         "message": "gesture confidence low source=driver type=palm_open confidence=0.60"}
        for _ in range(3)
    ]
    assert "gesture_low_conf" in _ids(logs)


# ═══════════════════════════════════════════════════════════════
# GestureFalseTriggerRule — 手势误触发率
# ═══════════════════════════════════════════════════════════════

def test_gesture_false_trigger():
    """最近 10 条中 4 条 stable=false（40%）→ 触发"""
    logs = []
    for i in range(10):
        stable = "false" if i < 4 else "true"
        logs.append({
            "module": "gesture", "level": "INFO",
            "message": f"gesture event source=driver type=palm_open confidence=0.90 stable={stable} command=volume_up",
        })
    assert "gesture_false_trigger" in _ids(logs)


def test_gesture_false_trigger_below_threshold():
    """最近 10 条中 2 条 stable=false（20%）→ 不触发"""
    logs = []
    for i in range(10):
        stable = "false" if i < 2 else "true"
        logs.append({
            "module": "gesture", "level": "INFO",
            "message": f"gesture event source=driver type=palm_open confidence=0.90 stable={stable} command=volume_up",
        })
    assert "gesture_false_trigger" not in _ids(logs)


def test_gesture_false_trigger_not_enough_logs():
    """不足 10 条 → 不触发"""
    logs = [
        {"module": "gesture", "level": "INFO",
         "message": "gesture event source=driver type=palm_open confidence=0.90 stable=false command=volume_up"}
        for _ in range(8)
    ]
    assert "gesture_false_trigger" not in _ids(logs)
