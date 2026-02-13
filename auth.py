from fastapi import HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import secrets
import jwt
import httpx
import bcrypt
from datetime import datetime, timedelta
from pymongo import ReturnDocument
from config import BREVO_API_KEY, BREVO_SENDER_EMAIL, BREVO_EMAIL_ENDPOINT, JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRATION_MINUTES, logger
from database import get_users_collection

# --- Auth Request Models ---
class RegisterRequest(BaseModel):
    username: str
    email: str

class VerifyOtpRequest(BaseModel):
    email: str
    otp: str
    password: str

class LoginRequest(BaseModel):
    identifier: str
    password: str

# -------------------------
# AUTH HELPERS
# -------------------------
def _require_user_collection():
    users_collection = get_users_collection()
    if users_collection is None:
        logger.error("User store not configured; check MongoDB settings.")
        raise HTTPException(status_code=500, detail="User store not configured.")
    return users_collection

def _create_access_token(email: str) -> str:
    if not JWT_SECRET:
        logger.error("JWT secret missing; cannot issue tokens.")
        raise HTTPException(status_code=500, detail="Authentication service unavailable.")
    expiry = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    payload = {"email": email, "exp": expiry}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def _generate_otp() -> int:
    return secrets.randbelow(900000) + 100000

def _normalize_email(email: str) -> str:
    return email.strip().lower()

async def _send_otp_email(recipient: str, otp: int) -> None:
    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    payload = {
        "sender": {"email": BREVO_SENDER_EMAIL},
        "to": [{"email": recipient}],
        "subject": "Your Verification OTP",
        "htmlContent": f"<p>Your OTP is <strong>{otp}</strong></p>",
        "textContent": f"Your OTP is {otp}",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            BREVO_EMAIL_ENDPOINT,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

# -------------------------
# AUTH ENDPOINTS
# -------------------------
async def register_user(payload: RegisterRequest) -> Dict[str, str]:
    collection = _require_user_collection()
    if not BREVO_API_KEY:
        logger.error("BREVO_API_KEY missing; cannot send OTP emails.")
        raise HTTPException(status_code=500, detail="Email service unavailable.")

    email = _normalize_email(payload.email)
    username = payload.username.strip()

    existing_user = await collection.find_one({
        "email": email,
        "otp": {"$exists": False},
    })
    if existing_user:
        raise HTTPException(status_code=400, detail="User already registered")

    otp = _generate_otp()
    await collection.update_one(
        {"email": email},
        {"$set": {"username": username, "email": email, "otp": otp}},
        upsert=True,
    )

    try:
        await _send_otp_email(email, otp)
        logger.info("OTP sent to %s", email)
    except httpx.HTTPStatusError as exc:
        logger.error("Brevo API error while sending OTP to %s: %s", email, exc.response.text)
        raise HTTPException(status_code=500, detail="Error sending OTP") from exc
    except httpx.HTTPError as exc:
        logger.error("Failed to send OTP to %s: %s", email, exc)
        raise HTTPException(status_code=500, detail="Error sending OTP") from exc

    return {"message": "OTP sent to email"}


async def verify_otp(payload: VerifyOtpRequest) -> Dict[str, Any]:
    collection = _require_user_collection()
    email = _normalize_email(payload.email)

    temp_user = await collection.find_one({"email": email})
    if not temp_user or "otp" not in temp_user:
        raise HTTPException(status_code=404, detail="No OTP record")

    try:
        submitted_otp = int(payload.otp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid OTP") from exc

    if temp_user["otp"] != submitted_otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    hashed_password = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    updated_user = await collection.find_one_and_update(
        {"email": email},
        {"$set": {"password": hashed_password}, "$unset": {"otp": ""}},
        return_document=ReturnDocument.AFTER,
    )

    if not updated_user:
        raise HTTPException(status_code=500, detail="Server error")

    token = _create_access_token(updated_user["email"])

    return {
        "message": "Verified successfully",
        "token": token,
        "user": {
            "username": updated_user.get("username"),
            "email": updated_user.get("email"),
        },
    }


async def login_user(payload: LoginRequest) -> Dict[str, Any]:
    collection = _require_user_collection()
    identifier_raw = payload.identifier.strip()
    email_candidate = _normalize_email(identifier_raw)

    query_conditions = [
        {"username": identifier_raw},
        {"email": identifier_raw},
    ]
    if email_candidate != identifier_raw:
        query_conditions.append({"email": email_candidate})

    user = await collection.find_one(
        {
            "$or": query_conditions,
            "otp": {"$exists": False},
        }
    )
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    stored_password = user.get("password")
    if not stored_password:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not bcrypt.checkpw(payload.password.encode("utf-8"), stored_password.encode("utf-8")):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = _create_access_token(user["email"])

    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "username": user.get("username"),
            "email": user.get("email"),
        },
    }
