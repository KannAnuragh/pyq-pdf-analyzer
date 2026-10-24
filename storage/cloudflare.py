"""
Cloudflare R2 image uploader.

Uploads image crops to a Cloudflare R2 bucket via the S3-compatible API.
Returns a public CDN URL using the configured custom domain.

URL pattern: https://examcdn.trogon.info/qai/{year}/{month}/{day}/{pdf_slug}/{uuid}.png

Falls back to inline data:image/png;base64 strings if R2 is not configured.
"""

import os
import base64
import logging
import datetime
import uuid
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialised boto3 S3 client (created once, reused across all uploads)
# ---------------------------------------------------------------------------
_s3_client = None
_s3_init_failed = False


def _get_s3_client():
    """Returns a cached boto3 S3 client for R2, or None if unconfigured."""
    global _s3_client, _s3_init_failed

    if _s3_init_failed:
        return None
    if _s3_client is not None:
        return _s3_client

    account_id = (os.getenv("R2_ACCOUNT_ID", "") or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")).strip()
    access_key = (os.getenv("R2_ACCESS_KEY_ID", "") or os.getenv("CLOUDFLARE_ACCESS_KEY_ID", "")).strip()
    secret_key = (os.getenv("R2_SECRET_ACCESS_KEY", "") or os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY", "")).strip()

    # Reject placeholder strings
    if account_id.lower() in ("your_account_id", "your_account_id_here", "your_cloudflare_account_id"):
        account_id = ""
    if access_key.lower() in ("your_access_key", "your_access_key_id", "your_r2_access_key_id"):
        access_key = ""
    if secret_key.lower() in ("your_secret_key", "your_secret_access_key", "your_r2_secret_access_key"):
        secret_key = ""

    if not account_id or not access_key or not secret_key:
        logger.warning(
            "Cloudflare R2 credentials (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY) "
            "are missing or set to placeholder values. Image crops will fall back to base64 data URIs."
        )
        _s3_init_failed = True
        return None

    try:
        import boto3
        from botocore.config import Config

        _s3_client = boto3.client(
            service_name="s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(
                retries={"max_attempts": 2, "mode": "standard"},
                connect_timeout=5,
                read_timeout=10,
            ),
        )
        logger.info("R2 S3 client initialised successfully")
        return _s3_client
    except Exception as e:
        logger.error(f"Failed to initialise R2 S3 client: {e}")
        _s3_init_failed = True
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_image_to_cloudflare(image_bytes: bytes, filename: str = "unknown") -> str:
    """
    Uploads an image to Cloudflare R2 and returns the public CDN URL.

    URL format:
        https://examcdn.trogon.info/qai/{year}/{month}/{day}/{pdf_slug}/{uuid}.png

    If R2 is not configured or the upload fails, gracefully falls back
    to returning a data:image/png;base64 string.
    """
    s3 = _get_s3_client()
    bucket = (os.getenv("R2_BUCKET_NAME", "") or os.getenv("CLOUDFLARE_BUCKET_NAME", "") or "examportal").strip()

    if s3 is None or not bucket:
        logger.debug("R2 not available, falling back to base64")
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # Build the object key:  qai/2026/09/18/my-pdf-name/a3f8b1c9d2e4.png
    now = datetime.datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")

    # Sanitise the PDF filename into a URL-safe slug
    clean_filename = re.sub(r'[^a-zA-Z0-9_-]', '', filename)
    if not clean_filename:
        clean_filename = "doc"

    unique_id = uuid.uuid4().hex[:12]
    object_key = f"qai/{year}/{month}/{day}/{clean_filename}/{unique_id}.png"

    try:
        s3.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=image_bytes,
            ContentType="image/png",
        )

        custom_domain = (
            os.getenv("R2_CUSTOM_DOMAIN", "")
            or os.getenv("CLOUDFLARE_CUSTOM_DOMAIN", "")
            or "https://examcdn.trogon.info"
        ).strip().rstrip("/")
        cdn_url = f"{custom_domain}/{object_key}"
        logger.info(f"Uploaded to R2: {cdn_url}")
        return cdn_url

    except Exception as e:
        logger.error(f"R2 upload failed for key '{object_key}': {e}")
        print(f"[R2 Upload] Failed: {e}", flush=True)

    # Fallback to base64 if upload failed
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"
