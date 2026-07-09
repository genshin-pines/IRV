"""告警 API 错误输入不崩溃测试

用法：
  1. 先启动后端: uvicorn backend.main:app --reload --port 8000
  2. 运行测试:  python test_error_inputs.py

判据：
  - 必须拒绝的输入 → 返回 4xx + {"ok": false, ...}
  - 可忽略的输入   → 返回 2xx + {"ok": true, ...}（静默忽略，不崩溃即可）
  - 任何 5xx 或进程崩溃 → 不通过
"""

import requests
import sys

BASE = "http://localhost:8000"

# expect_reject=True:  这个输入必须被明确拒绝（4xx + ok:false）
# expect_reject=False: 这个输入可以被静默忽略（2xx + ok:true 也通过）
TEST_CASES = [
    # ── 告警列表（参数: level可选, hours:1-168, limit:1-500） ──
    ("hours为负数",              "GET",  "/api/alerts?hours=-5",         True),
    ("hours为非数字",            "GET",  "/api/alerts?hours=abc",        True),
    ("limit为负数",              "GET",  "/api/alerts?limit=-10",        True),
    ("limit为非数字",            "GET",  "/api/alerts?limit=abc",        True),
    ("空参数",                   "GET",  "/api/alerts",                  False),
    ("含特殊字符的level（忽略）","GET",  "/api/alerts?level=<script>",    False),

    # ── 告警详情（不存在的记录必须拒绝） ──
    ("不存在的告警ID",           "GET",  "/api/alerts/99999",    True),
    ("ID为非数字",               "GET",  "/api/alerts/abc",      True),
    ("ID为0",                    "GET",  "/api/alerts/0",        True),
    ("ID为负数",                 "GET",  "/api/alerts/-1",       True),

    # ── 确认告警 ──
    ("确认不存在的ID",           "POST", "/api/alerts/99999/acknowledge", True, '{}'),
    ("确认ID为非数字",           "POST", "/api/alerts/abc/acknowledge",   True, '{}'),
    ("确认ID为0",                "POST", "/api/alerts/0/acknowledge",     True, '{}'),

    # ── 告警统计（未知 query 参数被忽略，可接受 200） ──
    ("stats非法日期参数",        "GET",  "/api/alerts/stats?from=invalid&to=xxx", False),
    ("stats空参数",              "GET",  "/api/alerts/stats",                     False),

    # ── 日志查询（参数校验由 FastAPI ge=1/le=500 保证） ──
    ("日志n为非数字",            "GET",  "/api/logs?n=abc",   True),
    ("日志n为负数",              "GET",  "/api/logs?n=-10",   True),
    ("日志n为0",                 "GET",  "/api/logs?n=0",     True),
    ("日志n超大",                "GET",  "/api/logs?n=99999", True),
    ("日志module不存在于路由签名（静默忽略）", "GET", "/api/logs?module=" + "x" * 2000, False),
    ("日志SQL注入（未知参数被忽略）",        "GET", "/api/logs?module=';DROP TABLE--", False),
    ("日志XSS（未知参数被忽略）",            "GET", "/api/logs?module=<script>alert(1)</script>", False),

    # ── 模拟异常 ──
    ("simulate不存在的场景",     "POST", "/api/logs/simulate?scenario=no_such&count=10",     True),
    ("simulate负数count",        "POST", "/api/logs/simulate?scenario=plate_low_conf&count=-5", True),
    ("simulate超大count",        "POST", "/api/logs/simulate?scenario=mixed&count=100000",   True),
    ("simulate无参数使用默认值", "POST", "/api/logs/simulate",                               False),
    ("simulate空scenario",       "POST", "/api/logs/simulate?scenario=&count=10",            True),

    # ── 日志统计（未知 query 参数被忽略） ──
    ("logs/stats非法日期参数",   "GET",  "/api/logs/stats?from=xxx&to=yyy", False),
    ("logs/stats空参数",         "GET",  "/api/logs/stats",                False),

    # ── 不存在的路由 ──
    ("GET 不存在的路径",         "GET",  "/api/nonexistent",     True),
    ("POST 不存在的路径",        "POST", "/api/not_here",        True, '{}'),

    # ── WebSocket 用 HTTP 访问 ──
    ("ws用HTTP请求",             "GET",  "/ws/alerts",           True),

    # ── 极端输入（未知参数被忽略，可接受 200） ──
    ("超长query（未知参数）",    "GET",  "/api/logs?module=" + "x" * 5000, False),
    ("超大body到acknowledge",    "POST", "/api/alerts/1/acknowledge",     False,
     '{"x":"' + "y" * 10000 + '"}'),
]


def run():
    print("=" * 60)
    print("告警 API 错误输入不崩溃测试")
    print(f"目标: {BASE}")
    print(f"测试用例: {len(TEST_CASES)} 项")
    print("=" * 60)

    passed = 0
    benign = 0   # 可忽略输入，返回 200（正常）
    failed = 0
    crashed = False

    for case in TEST_CASES:
        desc = case[0]
        method = case[1]
        path = case[2]
        expect_reject = case[3]
        body = case[4] if len(case) > 4 else None
        url = BASE + path

        try:
            if method == "GET":
                resp = requests.get(url, timeout=10)
            else:
                kwargs = {"timeout": 10}
                if body is not None:
                    kwargs["data"] = body
                    if body.startswith("{") or body.startswith("["):
                        kwargs["headers"] = {"Content-Type": "application/json"}
                resp = requests.post(url, **kwargs)

            # ── 关键判据：绝不能 5xx ──
            if resp.status_code >= 500:
                print(f"  FAIL [{resp.status_code}] {desc}")
                print(f"       -> 服务端异常(5xx)，不应出现！")
                failed += 1
                continue

            # ── 解析响应 JSON ──
            try:
                data = resp.json()
                is_ok_false = isinstance(data, dict) and data.get("ok") is False
                is_ok_true = isinstance(data, dict) and data.get("ok") is True
            except ValueError:
                print(f"  FAIL [{resp.status_code}] {desc} - 返回非JSON")
                failed += 1
                continue

            # ── 判断逻辑 ──
            if 400 <= resp.status_code < 500 and is_ok_false:
                # 正确拒绝了
                print(f"  OK   [{resp.status_code}] {desc}")
                passed += 1
            elif 200 <= resp.status_code < 300 and is_ok_true and not expect_reject:
                # 可忽略的输入，静默处理了
                print(f"  OK   [{resp.status_code}] {desc} (safe ignore)")
                benign += 1
            elif expect_reject:
                # 期望被拒绝，但返回了 2xx 或格式不对
                print(f"  WARN [{resp.status_code}] {desc} - should be rejected")
                failed += 1
            else:
                # 意外的状态
                print(f"  WARN [{resp.status_code}] {desc} - unexpected format")
                failed += 1

        except requests.exceptions.ConnectionError:
            print(f"  CRASH! {desc}")
            print(f"        -> server process crashed or not started: {url}")
            crashed = True
            break
        except requests.exceptions.Timeout:
            print(f"  FAIL TIMEOUT {desc} - request timeout")
            failed += 1
        except Exception as e:
            print(f"  FAIL {desc} - client exception: {e}")
            failed += 1

    print()
    print("=" * 60)
    total_ok = passed + benign
    if crashed:
        print(f"CRASH! Server process died, test aborted.")
        print(f"  Passed {passed} (including {benign} safe-ignore), check backend code.")
        sys.exit(1)
    elif failed == 0:
        print(f"ALL PASS ({total_ok}/{len(TEST_CASES)})")
        print(f"  {passed} correctly rejected, {benign} safely ignored")
        print(f"  Zero 5xx errors, zero crashes.")
        sys.exit(0)
    else:
        print(f"  Passed {passed}, Safe-ignore {benign}, Failed {failed}")
        print(f"  {failed} items need fixing - check the WARN lines above.")
        sys.exit(1)


if __name__ == "__main__":
    run()
