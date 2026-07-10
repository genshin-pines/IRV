from __future__ import annotations

from datetime import datetime, timedelta
import base64
import hashlib
import hmac
import json
import os
import secrets
import uuid
from typing import Any
from urllib.parse import quote, urlencode

import requests
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import (
    AUTH_CODE_TTL_SEC,
    AUTH_SECRET_KEY,
    HUAWEI_SMS_APP_KEY,
    HUAWEI_SMS_APP_SECRET,
    HUAWEI_SMS_ENDPOINT,
    HUAWEI_SMS_SENDER,
    HUAWEI_SMS_SIGNATURE,
    HUAWEI_SMS_STATUS_CALLBACK,
    HUAWEI_SMS_TEMPLATE_ID,
    SMS_WEBHOOK_TOKEN,
    SMS_WEBHOOK_URL,
    WECHAT_APP_ID,
    WECHAT_APP_SECRET,
    WECHAT_REDIRECT_URI,
)
from backend.database import get_db
from backend.models.auth_user import AuthCode, AuthUser, WechatLoginState


router = APIRouter(prefix="/api/auth", tags=["auth"])


def _ok(data: Any = None, message: str = "ok") -> dict[str, Any]:
    return {"ok": True, "message": message, "data": data}


def _norm(value: str) -> str:
    return value.strip().lower()


def _role(value: str) -> str:
    return value if value in {"admin", "driver"} else "driver"


def _digest(value: str) -> str:
    return hmac.new(AUTH_SECRET_KEY.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def _password_hash(password: str, username: str, salt: str | None = None) -> str:
    salt = salt or base64.urlsafe_b64encode(os.urandom(16)).decode("ascii")
    raw = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), f"{username}:{salt}".encode("utf-8"), 160000)
    return f"pbkdf2_sha256${salt}${base64.urlsafe_b64encode(raw).decode('ascii')}"


def _verify_password(password: str, username: str, saved: str | None) -> bool:
    if not saved:
        return False
    try:
        method, salt, expected = saved.split("$", 2)
    except ValueError:
        return False
    if method != "pbkdf2_sha256":
        return False
    actual = _password_hash(password, username, salt).split("$", 2)[2]
    return hmac.compare_digest(actual, expected)


def _token(user: AuthUser) -> str:
    payload = {
        "sub": user.username,
        "role": user.role,
        "iat": int(datetime.utcnow().timestamp()),
        "nonce": secrets.token_urlsafe(8),
    }
    body = base64.urlsafe_b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii").rstrip("=")
    sig = hmac.new(AUTH_SECRET_KEY.encode("utf-8"), body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def _user_payload(user: AuthUser) -> dict[str, Any]:
    return {
        "username": user.username,
        "displayName": user.display_name or user.username,
        "role": user.role,
    }


def _session(user: AuthUser) -> dict[str, Any]:
    return {
        "user": _user_payload(user),
        "token": _token(user),
        "storage": "server-db: password/contact/openid hashed; browser session AES-GCM",
    }


def _huawei_sms_ready() -> bool:
    required = [
        HUAWEI_SMS_ENDPOINT,
        HUAWEI_SMS_APP_KEY,
        HUAWEI_SMS_APP_SECRET,
        HUAWEI_SMS_SENDER,
        HUAWEI_SMS_TEMPLATE_ID,
        HUAWEI_SMS_SIGNATURE,
    ]
    return all(i.strip() for i in required)


def _huawei_wsse_header() -> dict[str, str]:
    nonce = uuid.uuid4().hex
    created = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha256((nonce + created + HUAWEI_SMS_APP_SECRET).encode("utf-8")).digest()
    password_digest = base64.b64encode(digest).decode("ascii")
    return {
        "Authorization": 'WSSE realm="SDP",profile="UsernameToken",type="Appkey"',
        "X-WSSE": (
            f'UsernameToken Username="{HUAWEI_SMS_APP_KEY}",'
            f'PasswordDigest="{password_digest}",Nonce="{nonce}",Created="{created}"'
        ),
    }


def _send_huawei_sms(phone: str, code: str) -> None:
    body = {
        "from": HUAWEI_SMS_SENDER,
        "to": phone,
        "templateId": HUAWEI_SMS_TEMPLATE_ID,
        "templateParas": json.dumps([code], ensure_ascii=False),
        "signature": HUAWEI_SMS_SIGNATURE,
    }
    if HUAWEI_SMS_STATUS_CALLBACK:
        body["statusCallback"] = HUAWEI_SMS_STATUS_CALLBACK
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        **_huawei_wsse_header(),
    }
    try:
        response = requests.post(HUAWEI_SMS_ENDPOINT, data=urlencode(body), headers=headers, timeout=10)
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"华为云短信请求失败: {exc}") from exc
    if response.status_code >= 400:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"华为云短信发送失败: {response.text[:300]}")
    try:
        data = response.json()
    except ValueError:
        return
    result = data.get("result") or data.get("code") or ""
    if result and str(result) not in {"0", "000000"}:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"华为云短信发送失败: {data}")


def _find_or_create_contact_user(db: Session, contact: str, role: str) -> AuthUser:
    target_hash = _digest(_norm(contact))
    user = db.scalar(select(AuthUser).where(AuthUser.contact_hash == target_hash))
    if user:
        return user
    user = AuthUser(
        username=f"user_{target_hash[:10]}",
        display_name="验证码用户",
        role=_role(role),
        contact_hash=target_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _find_or_create_wechat_user(db: Session, openid: str, role: str) -> AuthUser:
    openid_hash = _digest(openid)
    user = db.scalar(select(AuthUser).where(AuthUser.wechat_openid_hash == openid_hash))
    if user:
        return user
    user = AuthUser(
        username=f"wx_{openid_hash[:10]}",
        display_name="微信用户",
        role=_role(role),
        wechat_openid_hash=openid_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class RegisterRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    display_name: str = Field("", alias="displayName", max_length=128)
    contact: str = Field("", max_length=128)
    role: str = "driver"

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        return _role(value)


class PasswordLoginRequest(BaseModel):
    username: str
    password: str


class SmsSendRequest(BaseModel):
    phone: str = Field(min_length=5, max_length=32)
    role: str = "driver"


class SmsLoginRequest(BaseModel):
    phone: str
    code: str = Field(min_length=4, max_length=8)
    role: str = "driver"


class WechatQrRequest(BaseModel):
    role: str = "driver"


@router.get("/security")
def security_report():
    return _ok(
        {
            "transport": "production should use HTTPS; login payloads are accepted only by backend auth endpoints",
            "password_storage": "PBKDF2-HMAC-SHA256 with per-user salt",
            "contact_storage": "HMAC-SHA256 digest, no plain phone/email stored",
            "browser_storage": "frontend session uses Web Crypto AES-GCM",
            "sms_provider": "huawei_cloud" if _huawei_sms_ready() else ("webhook" if SMS_WEBHOOK_URL else "development mode"),
            "wechat": "configured" if WECHAT_APP_ID and WECHAT_APP_SECRET else "development mode",
        }
    )


@router.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    username = _norm(payload.username)
    if db.scalar(select(AuthUser).where(AuthUser.username == username)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="账号已存在")
    contact_hash = _digest(_norm(payload.contact)) if payload.contact else None
    if contact_hash and db.scalar(select(AuthUser).where(AuthUser.contact_hash == contact_hash)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="手机号或邮箱已被绑定")
    user = AuthUser(
        username=username,
        password_hash=_password_hash(payload.password, username),
        display_name=payload.display_name.strip() or username,
        role=payload.role,
        contact_hash=contact_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _ok(_session(user), "注册成功")


@router.post("/login/password")
def login_password(payload: PasswordLoginRequest, db: Session = Depends(get_db)):
    username = _norm(payload.username)
    demo = {
        "admin": ("123456", "admin", "值班管理员"),
        "driver": ("123456", "driver", "车主用户"),
    }.get(username)
    user = db.scalar(select(AuthUser).where(AuthUser.username == username))
    if not user and demo and payload.password == demo[0]:
        user = AuthUser(username=username, password_hash=_password_hash(demo[0], username), role=demo[1], display_name=demo[2])
        db.add(user)
        db.commit()
        db.refresh(user)
    if not user or not _verify_password(payload.password, username, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="账号或密码错误")
    return _ok(_session(user), "登录成功")


@router.post("/sms/send")
def sms_send(payload: SmsSendRequest, db: Session = Depends(get_db)):
    phone = _norm(payload.phone)
    code = f"{secrets.randbelow(900000) + 100000}"
    expires_at = datetime.utcnow() + timedelta(seconds=AUTH_CODE_TTL_SEC)
    auth_code = AuthCode(
        target_hash=_digest(phone),
        code_hash=_digest(f"{phone}:{code}"),
        role=_role(payload.role),
        expires_at=expires_at,
    )
    db.add(auth_code)
    db.commit()
    dev_mode = not (_huawei_sms_ready() or SMS_WEBHOOK_URL)
    provider = "development"
    if _huawei_sms_ready():
        _send_huawei_sms(phone, code)
        provider = "huawei_cloud"
    elif SMS_WEBHOOK_URL:
        headers = {"Authorization": f"Bearer {SMS_WEBHOOK_TOKEN}"} if SMS_WEBHOOK_TOKEN else {}
        try:
            requests.post(SMS_WEBHOOK_URL, json={"to": phone, "code": code, "scene": "IRV_LOGIN"}, headers=headers, timeout=8)
        except requests.RequestException as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"短信服务发送失败: {exc}") from exc
        provider = "webhook"
    return _ok(
        {
            "sent": True,
            "dev_mode": dev_mode,
            "provider": provider,
            "dev_code": code if dev_mode else None,
            "expires_in": AUTH_CODE_TTL_SEC,
        },
        "验证码已发送",
    )


@router.post("/login/sms")
def login_sms(payload: SmsLoginRequest, db: Session = Depends(get_db)):
    phone = _norm(payload.phone)
    target_hash = _digest(phone)
    code_hash = _digest(f"{phone}:{payload.code.strip()}")
    auth_code = db.scalar(
        select(AuthCode)
        .where(AuthCode.target_hash == target_hash, AuthCode.code_hash == code_hash, AuthCode.used.is_(False))
        .order_by(AuthCode.created_at.desc())
    )
    if not auth_code or auth_code.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="验证码无效或已过期")
    auth_code.used = True
    user = _find_or_create_contact_user(db, phone, auth_code.role or payload.role)
    db.commit()
    return _ok(_session(user), "验证码登录成功")


@router.post("/wechat/qrcode")
def wechat_qrcode(payload: WechatQrRequest, db: Session = Depends(get_db)):
    state = secrets.token_urlsafe(24)
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    if WECHAT_APP_ID:
        redirect_uri = quote(WECHAT_REDIRECT_URI, safe="")
        qr_url = (
            "https://open.weixin.qq.com/connect/qrconnect"
            f"?appid={WECHAT_APP_ID}&redirect_uri={redirect_uri}&response_type=code"
            f"&scope=snsapi_login&state={state}#wechat_redirect"
        )
        dev_mode = False
    else:
        qr_url = f"irv-wechat-login://scan?state={state}"
        dev_mode = True
    item = WechatLoginState(state=state, role=_role(payload.role), qr_url=qr_url, expires_at=expires_at)
    db.add(item)
    db.commit()
    return _ok({"state": state, "qr_url": qr_url, "dev_mode": dev_mode, "expires_in": 600})


@router.get("/wechat/status/{state}")
def wechat_status(state: str, db: Session = Depends(get_db)):
    item = db.get(WechatLoginState, state)
    if not item or item.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="二维码已失效")
    data: dict[str, Any] = {"state": state, "status": item.status}
    if item.status == "confirmed" and item.user_id:
        user = db.get(AuthUser, item.user_id)
        if user:
            data.update(_session(user))
    return _ok(data)


@router.post("/wechat/dev-confirm/{state}")
def wechat_dev_confirm(state: str, db: Session = Depends(get_db)):
    item = db.get(WechatLoginState, state)
    if not item or item.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="二维码已失效")
    if WECHAT_APP_ID and WECHAT_APP_SECRET:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="已配置真实微信，不允许开发确认")
    user = _find_or_create_wechat_user(db, f"dev-openid-{state}", item.role)
    item.status = "confirmed"
    item.user_id = user.id
    db.commit()
    return _ok(_session(user), "开发模式扫码确认成功")


@router.get("/wechat/callback", response_class=HTMLResponse)
def wechat_callback(code: str = Query(...), state: str = Query(...), db: Session = Depends(get_db)):
    item = db.get(WechatLoginState, state)
    if not item or item.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="二维码已失效")
    if not WECHAT_APP_ID or not WECHAT_APP_SECRET:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="微信密钥未配置")
    url = (
        "https://api.weixin.qq.com/sns/oauth2/access_token"
        f"?appid={WECHAT_APP_ID}&secret={WECHAT_APP_SECRET}&code={code}&grant_type=authorization_code"
    )
    try:
        result = requests.get(url, timeout=8).json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"微信接口调用失败: {exc}") from exc
    openid = result.get("openid")
    if not openid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=result.get("errmsg", "微信授权失败"))
    user = _find_or_create_wechat_user(db, openid, item.role)
    item.status = "confirmed"
    item.user_id = user.id
    db.commit()
    return HTMLResponse("<h2>微信授权成功</h2><p>请回到 IRV 页面完成登录。</p>")
