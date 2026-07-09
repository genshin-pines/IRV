# IRV Agent Alert API

## REST

- `GET /api/alerts`: Alert history with pagination and filters.
- `GET /api/alerts/{alert_id}`: Alert detail.
- `GET /api/alerts/stats`: Today count, unread count, ERROR count, CRITICAL count, module distribution and four-hour trend.
- `POST /api/alerts/{alert_id}/acknowledge`: Acknowledge an alert.
- `DELETE /api/alerts/{alert_id}`: Delete an alert.
- `GET /api/logs`: Recent logs with pagination, module, level and time filters.
- `GET /api/logs/stats`: Log counts by level and module.
- `POST /api/logs/simulate`: Generate demo logs for `plate_low_conf`, `camera_disconnect`, `gesture_jitter`, `api_timeout`, `login_fail` and `mixed`.

All REST APIs return:

```json
{
  "ok": true,
  "data": {},
  "message": "success",
  "trace_id": "20260708-xxxxxxxx"
}
```

## WebSocket

Endpoint: `ws://127.0.0.1:8000/ws/alerts`

New alerts are broadcast to every connected client:

```json
{
  "id": 20,
  "level": "ERROR",
  "title": "摄像头连接中断",
  "summary": "摄像头连接中断：请检查摄像头供电、RTSP 地址和网络链路。",
  "source_module": "camera",
  "created_at": "2026-07-08T02:49:38.329535"
}
```
