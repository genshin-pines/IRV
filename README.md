# IRV Agent Alert Module

This module provides the IRV log monitoring and intelligent alert backend.

## Run

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Swagger: http://localhost:8000/docs

## Main APIs

- `GET /api/alerts`
- `GET /api/alerts/{id}`
- `GET /api/alerts/stats`
- `POST /api/alerts/{id}/acknowledge`
- `DELETE /api/alerts/{id}`
- `GET /api/logs`
- `GET /api/logs/stats`
- `POST /api/logs/simulate`
- `WS /ws/alerts`

## Simulate

```bash
curl -X POST http://localhost:8000/api/logs/simulate \
  -H "Content-Type: application/json" \
  -d "{\"scenario\":\"mixed\",\"count\":10}"
```

The background agent scans new logs every 3 seconds, applies rules, writes alerts to SQLite, summarizes with DeepSeek when configured, and broadcasts new alerts through WebSocket.

## LLM

Environment variables:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `LLM_MODEL`

When the API key is missing or the provider returns 401/403/429/5xx/timeout/network errors, the agent falls back to rule-based summaries.
