from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import AUTH_SECRET_KEY
from backend.database import get_db
from backend.models.auth_user import AuthUser
from backend.schemas.common import ok
from backend.services import custom_gesture_service as service


router = APIRouter(prefix="/api/gesture-custom", tags=["gesture-custom"])


def get_authenticated_driver(
    authorization: str | None = Header(default=None), db: Session = Depends(get_db),
) -> AuthUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    try:
        token = authorization[7:]
        body_encoded, signature = token.rsplit(".", 1)
        expected = hmac.new(AUTH_SECRET_KEY.encode("utf-8"), body_encoded.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError("invalid signature")
        payload = json.loads(base64.urlsafe_b64decode(body_encoded + "==").decode("utf-8"))
        user = db.scalar(select(AuthUser).where(AuthUser.username == payload.get("sub", "")))
        if user is None or not user.is_active:
            raise ValueError("inactive user")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录状态无效") from exc
    if user.role != "driver":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="仅车主可管理自定义手势")
    return user


def _service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


class CaptureStartRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=64)
    action_code: str


class ImageConfirmRequest(BaseModel):
    draft_id: str = Field(min_length=8, max_length=64)


class BindingUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=64)
    action_code: str | None = None
    enabled: bool | None = None


@router.get("/catalog")
def api_catalog(user: AuthUser = Depends(get_authenticated_driver)):
    return ok(service.catalog())


@router.get("/bindings")
def api_list_bindings(user: AuthUser = Depends(get_authenticated_driver), db: Session = Depends(get_db)):
    return ok(service.list_bindings(db, user.id))


@router.post("/drafts/image")
async def api_image_draft(
    display_name: str = Form(...),
    action_code: str = Form(...),
    file: UploadFile = File(...),
    user: AuthUser = Depends(get_authenticated_driver),
):
    try:
        return ok(service.create_image_draft(user.id, display_name, action_code, await file.read()))
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/drafts/image/confirm")
def api_confirm_image_draft(
    payload: ImageConfirmRequest,
    user: AuthUser = Depends(get_authenticated_driver),
    db: Session = Depends(get_db),
):
    try:
        return ok(service.confirm_image_draft(db, user.id, payload.draft_id), "绑定成功")
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/camera-sessions")
def api_create_camera_session(payload: CaptureStartRequest, user: AuthUser = Depends(get_authenticated_driver)):
    try:
        return ok(service.create_camera_session(user.id, payload.display_name, payload.action_code))
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/camera-sessions/{session_id}/frame")
async def api_camera_frame(
    session_id: str,
    file: UploadFile = File(...),
    user: AuthUser = Depends(get_authenticated_driver),
    db: Session = Depends(get_db),
):
    try:
        return ok(service.process_camera_frame(db, user.id, session_id, await file.read()))
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/camera-sessions/{session_id}/active-frame")
def api_active_camera_frame(
    session_id: str,
    user: AuthUser = Depends(get_authenticated_driver),
    db: Session = Depends(get_db),
):
    try:
        return ok(service.process_active_stream_frame(db, user.id, session_id))
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        raise _service_error(exc) from exc


@router.post("/camera-sessions/{session_id}/confirm-first")
def api_confirm_camera_first(session_id: str, user: AuthUser = Depends(get_authenticated_driver)):
    try:
        return ok(service.confirm_camera_first(user.id, session_id))
    except Exception as exc:
        raise _service_error(exc) from exc


@router.patch("/bindings/{binding_id}")
def api_update_binding(
    binding_id: int,
    payload: BindingUpdateRequest,
    user: AuthUser = Depends(get_authenticated_driver),
    db: Session = Depends(get_db),
):
    try:
        return ok(service.update_binding(db, user.id, binding_id, payload.model_dump()))
    except Exception as exc:
        raise _service_error(exc) from exc


@router.delete("/bindings/{binding_id}")
def api_delete_binding(binding_id: int, user: AuthUser = Depends(get_authenticated_driver), db: Session = Depends(get_db)):
    try:
        service.delete_binding(db, user.id, binding_id)
        return ok({"deleted": True})
    except Exception as exc:
        raise _service_error(exc) from exc


