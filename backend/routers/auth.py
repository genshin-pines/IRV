from __future__ import annotations

from datetime import datetime, timedelta
import base64
import hashlib
import hmac
import json
import os
import secrets
import smtplib
from email.message import EmailMessage
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import (
    AUTH_CODE_COOLDOWN_SEC,
    AUTH_CODE_TTL_SEC,
    AUTH_SECRET_KEY,
    EMAIL_FROM,
    EMAIL_SMTP_HOST,
    EMAIL_SMTP_PASSWORD,
    EMAIL_SMTP_PORT,
    EMAIL_SMTP_USER,
)
from backend.database import get_db
from backend.models.auth_user import AuthCode, AuthUser
from backend.services.log_service import write_log


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
        "storage": "server-db: password/contact hashed; browser session AES-GCM",
    }


def _ensure_code_cooldown(db: Session, target_hash: str) -> None:
    latest = db.scalar(select(AuthCode).where(AuthCode.target_hash == target_hash).order_by(AuthCode.created_at.desc()))
    if not latest:
        return
    retry_at = latest.created_at + timedelta(seconds=AUTH_CODE_COOLDOWN_SEC)
    now = datetime.utcnow()
    if retry_at > now:
        retry_after = max(1, int((retry_at - now).total_seconds()))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"验证码发送太频繁，请 {retry_after} 秒后再试",
            headers={"Retry-After": str(retry_after)},
        )


def _email_ready() -> bool:
    return all(
        value.strip()
        for value in (EMAIL_SMTP_HOST, EMAIL_SMTP_USER, EMAIL_SMTP_PASSWORD, EMAIL_FROM)
    )


def _send_email_code(email: str, code: str) -> None:
    message = EmailMessage()
    message["Subject"] = "IRV 登录验证码"
    message["From"] = EMAIL_FROM
    message["To"] = email
    message.set_content(
        f"您的 IRV 登录验证码是：{code}\n\n"
        f"验证码 {AUTH_CODE_TTL_SEC // 60} 分钟内有效。若非本人操作，请忽略本邮件。"
    )
    try:
        if EMAIL_SMTP_PORT == 465:
            with smtplib.SMTP_SSL(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=10) as smtp:
                smtp.login(EMAIL_SMTP_USER, EMAIL_SMTP_PASSWORD)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=10) as smtp:
                smtp.starttls()
                smtp.login(EMAIL_SMTP_USER, EMAIL_SMTP_PASSWORD)
                smtp.send_message(message)
    except smtplib.SMTPException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"邮箱验证码发送失败: {exc}",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"邮箱服务器连接失败: {exc}",
        ) from exc


def _find_or_create_contact_user(db: Session, contact: str, role: str) -> AuthUser:
    target_hash = _digest(_norm(contact))
    user = db.scalar(select(AuthUser).where(AuthUser.contact_hash == target_hash))
    if user:
        requested_role = _role(role)
        if user.role != requested_role:
            user.role = requested_role
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


class CodeSendRequest(BaseModel):
    phone: str | None = Field(default=None, max_length=64)
    email: str | None = Field(default=None, max_length=128)
    role: str = "driver"


class CodeLoginRequest(BaseModel):
    phone: str | None = None
    email: str | None = None
    code: str = Field(min_length=4, max_length=8)
    role: str = "driver"


@router.get("/security")
def security_report():
    return _ok(
        {
            "transport": "production should use HTTPS",
            "password_storage": "PBKDF2-HMAC-SHA256 with per-user salt",
            "contact_storage": "HMAC-SHA256 digest, no plain phone/email stored",
            "browser_storage": "frontend session uses Web Crypto AES-GCM when available",
            "code_ttl_seconds": AUTH_CODE_TTL_SEC,
            "code_cooldown_seconds": AUTH_CODE_COOLDOWN_SEC,
            "email_provider": "smtp" if _email_ready() else "development",
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
    write_log("login", "INFO", f"register success user={username} role={user.role}")
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
        write_log("login", "WARNING", f"login fail user={username}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="账号或密码错误")
    write_log("login", "INFO", f"login success user={username} role={user.role}")
    return _ok(_session(user), "登录成功")


def _send_code(target: str, role: str, db: Session) -> dict[str, Any]:
    normalized = _norm(target)
    target_hash = _digest(normalized)
    _ensure_code_cooldown(db, target_hash)
    code = f"{secrets.randbelow(900000) + 100000}"
    db.add(
        AuthCode(
            target_hash=target_hash,
            code_hash=_digest(f"{normalized}:{code}"),
            role=_role(role),
            expires_at=datetime.utcnow() + timedelta(seconds=AUTH_CODE_TTL_SEC),
        )
    )
    db.commit()
    write_log("login", "INFO", f"auth code issued target_hash={target_hash[:10]} role={_role(role)}")
    return {"sent": True, "dev_mode": True, "provider": "development", "dev_code": code, "expires_in": AUTH_CODE_TTL_SEC}


@router.post("/sms/send")
def sms_send(payload: CodeSendRequest, db: Session = Depends(get_db)):
    if not payload.phone:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请输入手机号")
    return _ok(_send_code(payload.phone, payload.role, db), "验证码已发送")


@router.post("/email/send")
def email_send(payload: CodeSendRequest, db: Session = Depends(get_db)):
    email = _norm(payload.email or "")
    if not email or "@" not in email or email.startswith("@") or email.endswith("@"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="邮箱格式不正确")
    target_hash = _digest(email)
    _ensure_code_cooldown(db, target_hash)
    code = f"{secrets.randbelow(900000) + 100000}"
    dev_mode = not _email_ready()

    # Send first so a failed SMTP request does not create a cooldown record.
    if not dev_mode:
        _send_email_code(email, code)

    db.add(
        AuthCode(
            target_hash=target_hash,
            code_hash=_digest(f"{email}:{code}"),
            role=_role(payload.role),
            expires_at=datetime.utcnow() + timedelta(seconds=AUTH_CODE_TTL_SEC),
        )
    )
    db.commit()
    write_log(
        "login",
        "INFO",
        f"email auth code issued target_hash={target_hash[:10]} provider={'development' if dev_mode else 'smtp'}",
    )
    return _ok(
        {
            "sent": True,
            "dev_mode": dev_mode,
            "provider": "development" if dev_mode else "smtp",
            "dev_code": code if dev_mode else None,
            "expires_in": AUTH_CODE_TTL_SEC,
        },
        "邮箱验证码已发送",
    )


def _login_code(target: str, code: str, role: str, db: Session) -> dict[str, Any]:
    normalized = _norm(target)
    target_hash = _digest(normalized)
    code_hash = _digest(f"{normalized}:{code.strip()}")
    auth_code = db.scalar(
        select(AuthCode)
        .where(AuthCode.target_hash == target_hash, AuthCode.code_hash == code_hash, AuthCode.used.is_(False))
        .order_by(AuthCode.created_at.desc())
    )
    if not auth_code or auth_code.expires_at < datetime.utcnow():
        write_log("login", "WARNING", f"code login fail target_hash={target_hash[:10]}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="验证码无效或已过期")
    auth_code.used = True
    requested_role = _role(role)
    user = _find_or_create_contact_user(db, normalized, requested_role)
    db.commit()
    write_log(
        "login",
        "INFO",
        f"code login success user={user.username} role={user.role} issued_role={auth_code.role}",
    )
    return _session(user)


@router.post("/login/sms")
def login_sms(payload: CodeLoginRequest, db: Session = Depends(get_db)):
    if not payload.phone:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请输入手机号")
    return _ok(_login_code(payload.phone, payload.code, payload.role, db), "验证码登录成功")


@router.post("/login/email")
def login_email(payload: CodeLoginRequest, db: Session = Depends(get_db)):
    if not payload.email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请输入邮箱")
    return _ok(_login_code(payload.email, payload.code, payload.role, db), "邮箱验证码登录成功")
