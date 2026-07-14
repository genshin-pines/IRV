from alert_agent.rules import RuleEngine


def _titles(logs):
    return {result.title for result in RuleEngine().analyze(logs)}


def test_plate_low_conf_rule():
    logs = [
        {"module": "plate", "message": "plate confidence=0.60", "level": "WARNING"}
        for _ in range(5)
    ]
    assert any("车牌识别置信度持续偏低" in title for title in _titles(logs))


def test_camera_disconnect_rule():
    logs = [
        {"module": "camera", "message": "Broken pipe", "level": "ERROR"},
        {"module": "camera", "message": "Camera timeout", "level": "ERROR"},
    ]
    assert "摄像头连接中断" in _titles(logs)

    single = [{"module": "camera", "message": "Broken pipe", "level": "ERROR"}]
    assert "摄像头连接中断" not in _titles(single)


def test_gesture_jitter_rule():
    logs = [
        {"module": "gesture", "message": "gesture result=left", "level": "WARNING"},
        {"module": "gesture", "message": "gesture result=right", "level": "WARNING"},
        {"module": "gesture", "message": "gesture result=stop", "level": "WARNING"},
        {"module": "gesture", "message": "gesture result=left", "level": "WARNING"},
        {"module": "gesture", "message": "gesture result=right", "level": "WARNING"},
    ]
    assert "手势识别频繁跳变" in _titles(logs)


def test_api_timeout_rule():
    logs = [{"module": "backend", "message": "OCR request elapsed=9s", "level": "ERROR"}]
    assert "AI接口响应超时" in _titles(logs)


def test_login_fail_rule():
    logs = [
        {"module": "login", "message": "login fail user=admin", "level": "WARNING"}
        for _ in range(5)
    ]
    assert "连续登录失败" in _titles(logs)


def test_mixed_rule():
    logs = [
        {"module": "camera", "message": "RTSP disconnected", "level": "ERROR"},
        {"module": "camera", "message": "Camera timeout", "level": "ERROR"},
        {"module": "backend", "message": "LLM request timeout elapsed=9s", "level": "ERROR"},
    ]
    assert "系统存在复合异常" in _titles(logs)
