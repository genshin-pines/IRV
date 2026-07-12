from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import AUTH_SECRET_KEY
from backend.database import get_db
from backend.models.auth_user import AuthUser
from backend.services.preference_service import batch_set_preferences, delete_preference, get_all_preferences, get_preference, set_preference


router = APIRouter(prefix="/api/preferences", tags=["preferences"])


def response(data=None, message: str = "success", ok: bool = True) -> dict:
    return {"ok": ok, "data": data, "message": message, "trace_id": datetime.now().strftime("%Y%m%d-") + uuid4().hex[:8]}


class SetPreferenceRequest(BaseModel):
    value: str


class BatchPreferenceRequest(BaseModel):
    prefs: dict[str, str]


def get_current_user_id(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> int:
    """Extract user_id from Authorization Bearer token. Fallback to user_id=1 for demo mode."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        try:
            body_encoded, sig = token.rsplit(".", 1)
            expected_sig = hmac.new(
                AUTH_SECRET_KEY.encode("utf-8"), body_encoded.encode("ascii"), hashlib.sha256
            ).hexdigest()
            if hmac.compare_digest(expected_sig, sig):
                body = base64.urlsafe_b64decode(body_encoded + "==").decode("utf-8")
                payload = json.loads(body)
                username = payload.get("sub", "")
                if username:
                    user = db.scalar(select(AuthUser).where(AuthUser.username == username))
                    if user:
                        return user.id
        except Exception:
            pass
    # Demo fallback: return user_id=1 (the default driver account)
    return 1


@router.get("")
def api_get_all(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    prefs = get_all_preferences(db, user_id)
    return response(prefs)


@router.get("/{key}")
def api_get(key: str, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    value = get_preference(db, user_id, key)
    if value is None:
        raise HTTPException(status_code=404, detail="preference not found")
    return response({"key": key, "value": value})


@router.put("/{key}")
def api_set(key: str, payload: SetPreferenceRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    row = set_preference(db, user_id, key, payload.value)
    return response({"key": row.key, "value": row.value})


@router.post("/batch")
def api_batch(payload: BatchPreferenceRequest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    results = batch_set_preferences(db, user_id, payload.prefs)
    return response({row.key: row.value for row in results})


@router.delete("/{key}")
def api_delete(key: str, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    if not delete_preference(db, user_id, key):
        raise HTTPException(status_code=404, detail="preference not found")
    return response({"deleted": True})
