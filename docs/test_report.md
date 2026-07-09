# IRV Agent Alert Test Report

Validated areas:

- Rule engine tests for `plate_low_conf`, `camera_disconnect`, `gesture_jitter`, `api_timeout`, `login_fail` and `mixed`.
- API tests for health, simulate logs, alert list/detail/ACK/delete, alert stats and log filters.
- WebSocket ping and multi-client broadcast.
- LLM fallback without API key and on provider throttling/error.
- Database alert creation through SQLAlchemy service.

Latest local command:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected result:

```text
tests passed
```
