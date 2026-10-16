import hashlib
import secrets
import hmac
import time
import os
import ipaddress
import threading
import re
from datetime import datetime
from typing import Optional
from fastapi import Security, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from ..storage import database, models

API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

# Secret used for HMAC signed image URLs (never query-param api_key)
IMAGE_SIGNING_SECRET = os.getenv("IMAGE_SIGNING_SECRET", "qscan_image_signature_trogon_secret_key_2026")

# In-memory rate limiting structures
_rate_limit_lock = threading.Lock()
_key_request_history: dict[str, list[float]] = {}
_client_llm_history: dict[str, list[float]] = {}

DEFAULT_KEY_RATE_LIMIT = int(os.getenv("DEFAULT_KEY_RATE_LIMIT_PER_MIN", 120))
DEFAULT_LLM_RATE_LIMIT = int(os.getenv("DEFAULT_LLM_RATE_LIMIT_PER_MIN", 20))


def check_request_rate_limit(key_id: str, max_requests: int = DEFAULT_KEY_RATE_LIMIT, window_seconds: int = 60) -> None:
    """Sliding-window in-memory rate limiter per API key."""
    now = time.time()
    with _rate_limit_lock:
        history = _key_request_history.get(key_id, [])
        history = [t for t in history if now - t < window_seconds]
        if len(history) >= max_requests:
            retry_after = int(window_seconds - (now - history[0])) if history else window_seconds
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({max_requests} requests per minute). Try again in {max(1, retry_after)} seconds.",
                headers={"Retry-After": str(max(1, retry_after))}
            )
        history.append(now)
        _key_request_history[key_id] = history


def check_llm_rate_limit(client_id: str, max_requests: int = DEFAULT_LLM_RATE_LIMIT, window_seconds: int = 60) -> None:
    """Sliding-window in-memory rate limiter for expensive LLM routes per client."""
    now = time.time()
    with _rate_limit_lock:
        history = _client_llm_history.get(client_id, [])
        history = [t for t in history if now - t < window_seconds]
        if len(history) >= max_requests:
            retry_after = int(window_seconds - (now - history[0])) if history else window_seconds
            raise HTTPException(
                status_code=429,
                detail=f"LLM request rate limit reached ({max_requests} requests per minute). Please slow down.",
                headers={"Retry-After": str(max(1, retry_after))}
            )
        history.append(now)
        _client_llm_history[client_id] = history


def verify_internal_network(request: Request) -> None:
    """
    If RESTRICT_LEGACY_TO_INTERNAL is set to 'true', validates that the incoming
    request originates from a private IP (RFC 1918) or loopback address.
    """
    restrict = os.getenv("RESTRICT_LEGACY_TO_INTERNAL", "false").lower() in ("true", "1", "yes")
    if not restrict:
        return

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip_str = forwarded.split(",")[0].strip()
    elif request.client and request.client.host:
        client_ip_str = request.client.host
    else:
        client_ip_str = "127.0.0.1"

    try:
        ip = ipaddress.ip_address(client_ip_str)
        if not (ip.is_private or ip.is_loopback):
            raise HTTPException(
                status_code=403,
                detail="Access forbidden: Network restricted to internal callers."
            )
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Access forbidden: Invalid client IP address."
        )


def hash_api_key(raw_key: str) -> str:
    """Hashes an API key using SHA-256."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def generate_api_key(prefix: str = "qscan_live_") -> tuple[str, str, str]:
    """
    Generates a secure cryptographically random API key.
    Returns: (raw_key, key_hash, prefix)
    """
    token = secrets.token_urlsafe(32)
    raw_key = f"{prefix}{token}"
    key_hash = hash_api_key(raw_key)
    key_prefix = raw_key[:14]
    return raw_key, key_hash, key_prefix


IMAGE_ROUTE_PATTERN = re.compile(r"/pdfs/([^/]+)/questions/(\d+)/image")


def get_current_client(
    request: Request,
    x_api_key: Optional[str] = Security(api_key_header),
    db: Session = Depends(database.get_db)
) -> models.Client:
    """
    Strictly validates authentication (Header ONLY: X-API-Key, no query string).
    Rejects any request attempting to use ?api_key= in query parameters.
    Supports short-lived HMAC-SHA256 signed URLs specifically for /image routes.
    Checks that the key is active AND its client account is active.
    Compares hash using constant-time comparison (hmac.compare_digest).
    Throttles last_used_at update to at most once per 60 seconds.
    Enforces per-key request rate limits.
    """
    # 1. Allow ?api_key= ONLY on static media routes (images/PDF file downloads) where browser <img> cannot send custom headers
    is_media_route = bool(IMAGE_ROUTE_PATTERN.search(request.url.path) or request.url.path.endswith("/file"))
    if "api_key" in request.query_params and not is_media_route:
        raise HTTPException(
            status_code=401,
            detail="Authentication via query string '?api_key=' is removed. Please provide the 'X-API-Key' HTTP header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    if is_media_route and ("api_key" in request.query_params) and (not x_api_key or not x_api_key.strip()):
        x_api_key = request.query_params["api_key"].strip()

    # 2. Check for short-lived HMAC signed URL on image endpoints
    if not x_api_key or not x_api_key.strip():
        img_match = IMAGE_ROUTE_PATTERN.search(request.url.path)
        if img_match:
            job_id = img_match.group(1)
            q_num = int(img_match.group(2))
            expires = request.query_params.get("expires")
            sig = request.query_params.get("sig")
            part_str = request.query_params.get("part")
            part = int(part_str) if part_str is not None and part_str.isdigit() else None

            if expires and sig:
                try:
                    exp = int(expires)
                    if not verify_signed_image_token(job_id, q_num, part, exp, sig):
                        raise HTTPException(status_code=401, detail="Invalid or expired image signature.")
                    job = db.query(models.Job).filter(models.Job.id == job_id).first()
                    if not job:
                        raise HTTPException(status_code=404, detail="Job not found")
                    client = db.query(models.Client).filter(
                        models.Client.id == job.client_id,
                        models.Client.is_active == True
                    ).first()
                    if not client:
                        raise HTTPException(status_code=401, detail="Client account inactive or not found.")
                    return client
                except ValueError:
                    raise HTTPException(status_code=401, detail="Invalid image signature format.")

        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please provide the 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    raw_key = x_api_key.strip()
    key_hash = hash_api_key(raw_key)

    api_key_record = db.query(models.ApiKey).filter(
        models.ApiKey.key_hash == key_hash,
        models.ApiKey.is_active == True
    ).first()

    if not api_key_record:
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked API key.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # Constant-time comparison
    if not hmac.compare_digest(key_hash, api_key_record.key_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked API key.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    key_id = str(api_key_record.id)
    client_id = str(api_key_record.client_id)
    last_used = api_key_record.last_used_at

    client = db.query(models.Client).filter(
        models.Client.id == client_id,
        models.Client.is_active == True
    ).first()

    if not client:
        raise HTTPException(
            status_code=401,
            detail="Client account associated with this API key is inactive or not found.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # Throttle last_used_at update (at most once every 60 seconds)
    now_dt = datetime.utcnow()
    if not last_used or (now_dt - last_used).total_seconds() >= 60:
        try:
            db.query(models.ApiKey).filter(models.ApiKey.id == key_id).update(
                {"last_used_at": now_dt}
            )
            db.commit()
        except Exception:
            db.rollback()

    # Enforce request rate limit per key
    check_request_rate_limit(key_id)

    return client


def get_current_admin_client(
    client: models.Client = Depends(get_current_client)
) -> models.Client:
    """Requires the authenticated client to have is_admin=True privileges."""
    if not client.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Administrative privileges required for this action."
        )
    return client


# --- Signed Image URL Utilities (Phase 2) ---

def generate_signed_image_token(job_id: str, question_number: int, part: Optional[int], expires_at: int) -> str:
    """Generates an HMAC-SHA256 signature for short-lived image access."""
    payload = f"{job_id}:{question_number}:{part or 0}:{expires_at}"
    return hmac.new(IMAGE_SIGNING_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def verify_signed_image_token(job_id: str, question_number: int, part: Optional[int], expires: int, sig: str) -> bool:
    """Verifies that an image signature is valid and has not expired."""
    if time.time() > expires:
        return False
    expected = generate_signed_image_token(job_id, question_number, part, expires)
    return hmac.compare_digest(sig, expected)


def create_client_and_key(
    db: Session,
    name: str,
    daily_upload_limit: Optional[int] = None,
    is_admin: bool = False,
    prefix: str = "qscan_live_"
) -> tuple[models.Client, str]:
    """Helper function to onboard a new client and generate their raw API key."""
    client = models.Client(name=name, daily_upload_limit=daily_upload_limit, is_admin=is_admin, is_active=True)
    db.add(client)
    db.commit()
    db.refresh(client)

    raw_key, key_hash, key_prefix = generate_api_key(prefix=prefix)
    api_key_entry = models.ApiKey(
        client_id=client.id,
        key_hash=key_hash,
        prefix=key_prefix,
        name=f"Key for {name}",
        is_active=True
    )
    db.add(api_key_entry)
    db.commit()
    db.refresh(api_key_entry)

    return client, raw_key
