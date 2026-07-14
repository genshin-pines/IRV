from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select

from backend.database import SessionLocal, init_db
from backend.models.auth_user import AuthCode, AuthUser
from backend.routers import auth


def _cleanup(db, email: str) -> None:
    target_hash = auth._digest(auth._norm(email))
    db.execute(delete(AuthCode).where(AuthCode.target_hash == target_hash))
    db.execute(delete(AuthUser).where(AuthUser.contact_hash == target_hash))
    db.commit()


def test_smtp_email_code_can_log_in(monkeypatch):
    init_db()
    email = f"irv-{uuid4().hex}@example.com"
    sent = {}
    monkeypatch.setattr(auth, "_email_ready", lambda: True)
    monkeypatch.setattr(auth, "_send_email_code", lambda target, code: sent.update(email=target, code=code))

    with SessionLocal() as db:
        try:
            issued = auth.email_send(auth.CodeSendRequest(email=email, role="driver"), db)
            assert issued["data"]["provider"] == "smtp"
            assert issued["data"]["dev_code"] is None
            assert sent["email"] == email

            session = auth.login_email(
                auth.CodeLoginRequest(email=email, code=sent["code"], role="driver"),
                db,
            )
            assert session["data"]["user"]["role"] == "driver"
            assert session["data"]["token"]
        finally:
            _cleanup(db, email)


def test_verified_code_updates_existing_contact_role(monkeypatch):
    init_db()
    email = f"irv-{uuid4().hex}@example.com"
    sent = {}
    monkeypatch.setattr(auth, "_email_ready", lambda: True)
    monkeypatch.setattr(auth, "_send_email_code", lambda target, code: sent.update(email=target, code=code))

    with SessionLocal() as db:
        try:
            target_hash = auth._digest(auth._norm(email))
            db.add(
                AuthUser(
                    username=f"user_{target_hash[:10]}",
                    display_name="验证码用户",
                    role="driver",
                    contact_hash=target_hash,
                )
            )
            db.commit()

            auth.email_send(auth.CodeSendRequest(email=email, role="admin"), db)
            session = auth.login_email(
                auth.CodeLoginRequest(email=email, code=sent["code"], role="admin"),
                db,
            )

            assert session["data"]["user"]["role"] == "admin"
            assert db.scalar(select(AuthUser).where(AuthUser.contact_hash == target_hash)).role == "admin"
        finally:
            _cleanup(db, email)


def test_code_login_uses_role_selected_at_submit(monkeypatch):
    init_db()
    email = f"irv-{uuid4().hex}@example.com"
    sent = {}
    monkeypatch.setattr(auth, "_email_ready", lambda: True)
    monkeypatch.setattr(auth, "_send_email_code", lambda target, code: sent.update(email=target, code=code))

    with SessionLocal() as db:
        try:
            auth.email_send(auth.CodeSendRequest(email=email, role="driver"), db)
            session = auth.login_email(
                auth.CodeLoginRequest(email=email, code=sent["code"], role="admin"),
                db,
            )

            assert session["data"]["user"]["role"] == "admin"
        finally:
            _cleanup(db, email)


def test_failed_smtp_send_does_not_create_cooldown(monkeypatch):
    init_db()
    email = f"irv-{uuid4().hex}@example.com"
    target_hash = auth._digest(email)
    monkeypatch.setattr(auth, "_email_ready", lambda: True)

    def fail_send(_email, _code):
        raise HTTPException(status_code=502, detail="smtp unavailable")

    monkeypatch.setattr(auth, "_send_email_code", fail_send)
    with SessionLocal() as db:
        try:
            with pytest.raises(HTTPException, match="smtp unavailable"):
                auth.email_send(auth.CodeSendRequest(email=email, role="driver"), db)
            assert db.scalar(select(AuthCode).where(AuthCode.target_hash == target_hash)) is None
        finally:
            _cleanup(db, email)
