# IRV Agent Alert Deployment

## Environment

- Python 3.11+
- SQLite by default, switchable through `DATABASE_URL`
- DeepSeek/OpenAI-compatible LLM through environment variables:
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_BASE_URL`
  - `LLM_MODEL`

## Start

```powershell
cd E:\IRV\IRV-feature_lzh
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest --cov=backend --cov=alert_agent --cov-report=term-missing
```

The background scheduler starts with FastAPI and scans logs every 3 seconds.
