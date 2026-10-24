import os
import re
import json
import uuid
import datetime
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session
import pymupdf

from ..storage import database, models
from .auth import get_current_client, get_current_admin_client
from .routes import trigger_queue_worker
from pydantic import BaseModel
from ..models import schemas
from ..ingest.crop_utils import get_question_crop_rect, find_question_page
from .auth import get_current_client, get_current_admin_client, generate_api_key

import time
import threading

router = APIRouter(prefix="/v1", dependencies=[Depends(get_current_client)], tags=["v1"])

# Maximum upload size cap: 50MB by default, configurable via env
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", 50 * 1024 * 1024))



@router.post("/pdfs/upload", status_code=202)
async def upload_pdf(
    file: UploadFile = File(...),
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(database.get_db)
):
    """
    Stream upload a PDF file with size cap, magic byte validation, and daily limit check.
    Immediately returns job_id and 'pending' status.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename must be provided")

    # 1. Enforce daily upload cap per client if configured (default None/0 is unlimited)
    if client.daily_upload_limit and client.daily_upload_limit > 0:
        today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        uploaded_today = db.query(models.Job).filter(
            models.Job.client_id == client.id,
            models.Job.created_at >= today_start
        ).count()
        if uploaded_today >= client.daily_upload_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily upload limit reached ({client.daily_upload_limit} uploads per day). Please upgrade or try again tomorrow."
            )

    # 2. Setup isolated storage path: uploads/{client_id}/{job_id}.pdf
    job_id = str(uuid.uuid4())
    client_dir = os.path.join(database.UPLOADS_DIR, client.id)
    os.makedirs(client_dir, exist_ok=True)
    dest_path = os.path.join(client_dir, f"{job_id}.pdf")

    # 3. Stream upload chunk-by-chunk with size cap and %PDF- check
    total_bytes = 0
    first_chunk = True
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await file.read(65536)  # 64 KB chunks
                if not chunk:
                    break

                if first_chunk:
                    first_chunk = False
                    # Check %PDF- header (allowing leading whitespace or UTF-8 BOM)
                    header = chunk.lstrip(b"\xef\xbb\xbf \t\r\n")
                    if not header.startswith(b"%PDF-"):
                        f.close()
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid file format: File must be a valid PDF starting with %PDF-"
                        )

                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    f.close()
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum upload size limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB"
                    )

                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=500, detail=f"Failed to stream and save uploaded file: {e}")

    # 4. Insert Job into database with client_id
    job = models.Job(
        id=job_id,
        client_id=client.id,
        filename=os.path.basename(file.filename),
        file_path=dest_path,
        status="pending",
        stage="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # 5. Trigger the background queue worker
    trigger_queue_worker()

    return {
        "job_id": job.id,
        "status": "pending",
        "filename": job.filename
    }


@router.get("/pdfs/{job_id}/status")
def get_job_status(
    job_id: str,
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(database.get_db)
):
    """
    Check the current processing status of a job.
    Strictly scoped to the authenticated client (returns 404 for other clients).
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id,
        models.Job.client_id == client.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.id,
        "status": job.status,
        "filename": job.filename,
        "error": job.error_message
    }


@router.get("/pdfs/{job_id}/result")
def get_job_result(
    job_id: str,
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(database.get_db)
):
    """
    Retrieve the final structured JSON result for a completed job.
    Returns 409 Conflict with current status and error if job is not done.
    Strictly scoped to the authenticated client (returns 404 for other clients).
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id,
        models.Job.client_id == client.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "done":
        raise HTTPException(
            status_code=409,
            detail={
                "job_id": job.id,
                "status": job.status,
                "error": job.error_message
            }
        )

    if not job.result_json:
        raise HTTPException(status_code=500, detail="Job marked as done but result payload is missing")

    try:
        parsed_result = json.loads(job.result_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse result payload: {e}")

    return parsed_result


@router.get("/pdfs/{job_id}/questions/{question_number}/image")
def get_question_image(
    job_id: str,
    question_number: int,
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(database.get_db)
):
    """
    Serves the cropped image for a specific question with strict client ownership check.
    Returns 404 if the job belongs to another client.
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id,
        models.Job.client_id == client.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.result_json:
        raise HTTPException(status_code=404, detail="Question image not available (result pending)")

    # Check disk cache
    cache_dir = os.path.join(database.UPLOADS_DIR, client.id, "cache")
    cache_file = os.path.join(cache_dir, f"{job_id}_q{question_number}_p0.png")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as cf:
                cached_bytes = cf.read()
            return Response(
                content=cached_bytes,
                media_type="image/png",
                headers={
                    "Cache-Control": "private, no-cache, no-store, must-revalidate",
                    "X-Cache": "HIT"
                }
            )
        except Exception:
            pass

    file_path = job.file_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Source PDF file not found on disk")

    try:
        result_obj = schemas.PDFResult.model_validate_json(job.result_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decode job result: {e}")

    target_q_idx = -1
    for i, q in enumerate(result_obj.questions):
        if q.question_number == question_number:
            target_q_idx = i
            break

    if target_q_idx == -1:
        raise HTTPException(status_code=404, detail="Question not found")

    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open PDF: {e}")

    try:
        target_q = result_obj.questions[target_q_idx]
        hint_page = getattr(target_q, "page_number", None)
        target_stem = getattr(target_q, "stem", None)
        target_options = getattr(target_q, "options", None)
        target_page_idx = find_question_page(doc, question_number, hint_page, question_stem=target_stem)

        if target_page_idx is None:
            raise HTTPException(status_code=404, detail="Could not locate question in PDF pages")

        next_question_number = None
        next_question_stem = None
        end_page_idx = target_page_idx

        if target_q_idx + 1 < len(result_obj.questions):
            next_q = result_obj.questions[target_q_idx + 1]
            next_hint = getattr(next_q, "page_number", None)
            next_stem = getattr(next_q, "stem", None)
            next_page_idx = find_question_page(doc, next_q.question_number, next_hint, question_stem=next_stem)
            if next_page_idx is not None and next_page_idx >= target_page_idx:
                end_page_idx = next_page_idx
                next_question_number = next_q.question_number
                next_question_stem = next_stem

        pixmaps = []
        for p_idx in range(target_page_idx, end_page_idx + 1):
            page = doc[p_idx]
            is_cont = (p_idx > target_page_idx)
            p_next_q_num = next_question_number if p_idx == end_page_idx else None
            p_next_q_stem = next_question_stem if p_idx == end_page_idx else None

            crop_rect = get_question_crop_rect(
                page,
                question_number,
                p_next_q_num,
                p_next_q_stem,
                is_continuation=is_cont,
                current_question_stem=target_stem,
                current_options=target_options
            )
            pix = page.get_pixmap(clip=crop_rect, dpi=200) if crop_rect else page.get_pixmap(dpi=200)
            pixmaps.append(pix)

        if len(pixmaps) == 1:
            image_bytes = pixmaps[0].tobytes("png")
        else:
            from PIL import Image
            import io
            imgs = [Image.open(io.BytesIO(pix.tobytes("png"))) for pix in pixmaps]
            total_height = sum(img.height for img in imgs)
            max_width = max(img.width for img in imgs)

            stitched_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for img in imgs:
                stitched_img.paste(img, (0, y_offset))
                y_offset += img.height

            img_byte_arr = io.BytesIO()
            stitched_img.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()

        # Write to disk cache
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as cf:
                cf.write(image_bytes)
        except Exception as e:
            print(f"Warning: Failed to write cropped image cache: {e}")

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "private, no-cache, no-store, must-revalidate",
                "X-Image-Parts-Count": str(len(pixmaps))
            }
        )
    finally:
        doc.close()


@router.get("/pdfs")
def list_client_pdfs(
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(database.get_db)
):
    """
    List all uploaded PDFs and their processing statuses for the authenticated client.
    """
    jobs = db.query(models.Job).filter(
        models.Job.client_id == client.id
    ).order_by(models.Job.created_at.desc()).all()

    return [
        {
            "job_id": j.id,
            "filename": j.filename,
            "status": j.status,
            "stage": j.stage,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "error": j.error_message
        }
        for j in jobs
    ]


@router.delete("/pdfs/{job_id}")
def delete_client_pdf(
    job_id: str,
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(database.get_db)
):
    """
    Delete a job and its associated stored file.
    Strictly scoped to the authenticated client.
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id,
        models.Job.client_id == client.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.file_path and os.path.exists(job.file_path):
        try:
            os.remove(job.file_path)
        except Exception:
            pass

    # Clean up cached image crops
    cache_dir = os.path.join(database.UPLOADS_DIR, client.id, "cache")
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith(f"{job_id}_"):
                try:
                    os.remove(os.path.join(cache_dir, fname))
                except Exception:
                    pass

    db.delete(job)
    db.commit()

    return {"message": "Job successfully deleted", "job_id": job_id}


# --- Administrative Endpoints (Strictly requires Admin Client) ---

class AdminClientCreateRequest(BaseModel):
    name: str
    id: Optional[str] = None
    daily_upload_limit: Optional[int] = None
    is_admin: Optional[bool] = False

def _require_http_key_provisioning_allowed():
    """
    Restricts API key and client provisioning over HTTP.
    By default, only the backend administrator on the local host machine can create
    or manage API keys using the Admin CLI ('admin_cli.py').
    """
    allow = os.getenv("ALLOW_HTTP_KEY_PROVISIONING", "false").lower() in ("true", "1", "yes")
    if not allow:
        raise HTTPException(
            status_code=403,
            detail=(
                "Forbidden: Remote API key and client management is disabled over HTTP. "
                "Only the backend administrator on the host server can create, delete, or provision API keys "
                "using the Admin CLI ('./venv/bin/python -m pdf_qa_agent.scripts.admin_cli')."
            )
        )

@router.get("/admin/clients", tags=["v1_admin"])
def list_admin_clients(
    admin: models.Client = Depends(get_current_admin_client),
    db: Session = Depends(database.get_db)
):
    """List all registered clients in the multi-tenant system."""
    _require_http_key_provisioning_allowed()
    clients = db.query(models.Client).order_by(models.Client.created_at.desc()).all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "is_active": c.is_active,
            "daily_upload_limit": c.daily_upload_limit,
            "is_admin": c.is_admin,
            "created_at": c.created_at.isoformat() if c.created_at else None
        }
        for c in clients
    ]

@router.post("/admin/clients", status_code=201, tags=["v1_admin"])
def create_admin_client(
    req: AdminClientCreateRequest,
    admin: models.Client = Depends(get_current_admin_client),
    db: Session = Depends(database.get_db)
):
    """Create a new client account."""
    _require_http_key_provisioning_allowed()
    if req.id:
        existing = db.query(models.Client).filter(models.Client.id == req.id).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Client with ID '{req.id}' already exists.")

    client = models.Client(
        id=req.id or str(uuid.uuid4()),
        name=req.name,
        daily_upload_limit=req.daily_upload_limit,
        is_admin=req.is_admin or False,
        is_active=True
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    return {
        "id": client.id,
        "name": client.name,
        "daily_upload_limit": client.daily_upload_limit,
        "is_admin": client.is_admin,
        "is_active": client.is_active
    }

class AdminKeyCreateRequest(BaseModel):
    client_id: str
    name: Optional[str] = None
    raw_key: Optional[str] = None
    prefix: Optional[str] = "qscan_live_"
    create_client_if_missing: Optional[bool] = False

@router.get("/admin/keys", tags=["v1_admin"])
def list_admin_keys(
    client_id: Optional[str] = Query(None),
    admin: models.Client = Depends(get_current_admin_client),
    db: Session = Depends(database.get_db)
):
    """List API keys across all clients, or filter by specific client_id."""
    _require_http_key_provisioning_allowed()
    query = db.query(models.ApiKey)
    if client_id:
        query = query.filter(models.ApiKey.client_id == client_id)
    keys = query.order_by(models.ApiKey.created_at.desc()).all()
    return [
        {
            "id": k.id,
            "client_id": k.client_id,
            "prefix": k.prefix,
            "name": k.name,
            "is_active": k.is_active,
            "created_at": k.created_at.isoformat() if k.created_at else None,
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None
        }
        for k in keys
    ]

@router.post("/admin/keys", status_code=201, tags=["v1_admin"])
def create_or_add_admin_key(
    req: AdminKeyCreateRequest,
    admin: models.Client = Depends(get_current_admin_client),
    db: Session = Depends(database.get_db)
):
    """
    Generate a new API key or add/register a specified raw API key for a client.
    If raw_key is provided, it is hashed and stored; otherwise a cryptographically
    secure random key is generated.
    """
    _require_http_key_provisioning_allowed()
    client = db.query(models.Client).filter(models.Client.id == req.client_id).first()
    if not client:
        if req.create_client_if_missing:
            client = models.Client(
                id=req.client_id,
                name=req.name or f"Client {req.client_id}",
                daily_upload_limit=None,
                is_active=True
            )
            db.add(client)
            db.commit()
            db.refresh(client)
        else:
            raise HTTPException(status_code=404, detail=f"Client '{req.client_id}' not found.")

    import hashlib
    if req.raw_key and req.raw_key.strip():
        raw_key = req.raw_key.strip()
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        key_prefix = raw_key[:14]
    else:
        raw_key, key_hash, key_prefix = generate_api_key(prefix=req.prefix or "qscan_live_")

    existing_key = db.query(models.ApiKey).filter(models.ApiKey.key_hash == key_hash).first()
    if existing_key:
        if not existing_key.is_active:
            existing_key.is_active = True
            db.commit()
            return {
                "id": existing_key.id,
                "client_id": client.id,
                "name": existing_key.name,
                "prefix": existing_key.prefix,
                "raw_key": raw_key,
                "status": "re-enabled",
                "message": "Existing inactive key was re-enabled."
            }
        raise HTTPException(status_code=409, detail="An active key with this exact hash already exists.")

    key_record = models.ApiKey(
        client_id=client.id,
        key_hash=key_hash,
        prefix=key_prefix,
        name=req.name or f"Key for {client.name}",
        is_active=True
    )
    db.add(key_record)
    db.commit()
    db.refresh(key_record)

    return {
        "id": key_record.id,
        "client_id": client.id,
        "name": key_record.name,
        "prefix": key_prefix,
        "raw_key": raw_key,
        "created_at": key_record.created_at.isoformat() if key_record.created_at else None,
        "message": "Key created successfully. Store the raw key securely; it will not be displayed again."
    }

@router.delete("/admin/keys/{key_id}", tags=["v1_admin"])
def revoke_admin_key(
    key_id: str,
    admin: models.Client = Depends(get_current_admin_client),
    db: Session = Depends(database.get_db)
):
    """Revoke an API key by marking is_active=False."""
    _require_http_key_provisioning_allowed()
    key_rec = db.query(models.ApiKey).filter(models.ApiKey.id == key_id).first()
    if not key_rec:
        raise HTTPException(status_code=404, detail="API key not found.")
    key_rec.is_active = False
    db.commit()
    return {"message": f"API key '{key_id}' successfully revoked.", "id": key_id}
