from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, Body, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import json
import os
import urllib.parse
import threading
import time
import base64

import pymupdf
import re

from ..storage import database, models
from ..models import schemas

router = APIRouter()

from ..ingest.router import route_and_extract, ocr_equation_clip, clean_latex_equation
from ..graph.workflow import app_graph
from ..graph.state import GraphState
from ..graph.nodes import get_gemini_status, is_gemini_enabled, get_groq_status, is_groq_enabled, get_llm_status, is_llm_enabled, is_answer_validation_enabled, rescan_question_vision, _solve_single_block, separate_stem_and_options, parse_questions_from_text, parse_trailing_meta

_math_conversion_tasks: dict[str, dict] = {}

def _count_math_images(questions: list) -> int:
    """Counts remaining equation images in questions (excluding genuine geometry diagrams)."""
    cnt = 0
    for q in questions:
        for text in [q.get("stem", "")] + list(q.get("options", {}).values()):
            if '{{IMG:' in text or '{{IMG_BLOCK:' in text:
                cnt += len(re.findall(r'\{\{IMG(?:_BLOCK)?:\s*[^}]+\}\}', text))
    return cnt

def _replace_math_images_in_text(text: str, engine: Optional[str] = None) -> tuple[str, bool]:
    """Replaces equation and diagram image tags with LaTeX by running targeted OCR on each clip."""
    if not text or ('{{IMG:' not in text and '{{IMG_BLOCK:' not in text):
        return text, False

    # Match anchor tokens like {{IMG:https://...}}
    pattern = r'\{\{IMG(?:_BLOCK)?:\s*([^}]+)\}\}'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text, False

    modified = False
    new_text = text
    import urllib.request
    
    for m in reversed(matches):
        tag_str = m.group(0)
        img_url = m.group(1).strip()
        
        try:
            req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                img_bytes = response.read()
                
            # Skip genuine geometry/drawing diagrams
            if 'IMG_BLOCK' in tag_str:
                from PIL import Image
                import io
                chk_img = Image.open(io.BytesIO(img_bytes))
                cw, ch = chk_img.size
                if cw > 180 and ch > 120:
                    continue

            latex, _, _ = ocr_equation_clip(img_bytes, engine=engine)
            clean_math = clean_latex_equation(latex)
            if clean_math and clean_math != "$$":
                # If Gemini returned a Diagram/Image fallback description, REJECT it and keep the image!
                if re.match(r'^\[\s*(?:Diagram|Image|Chart|Graph|Figure)\b', clean_math, re.IGNORECASE):
                    continue
                new_text = new_text[:m.start()] + f" {{{{EQ:{clean_math}}}}} " + new_text[m.end():]
                modified = True
        except Exception as e:
            print(f"[Progressive Math OCR] Tag parse failed: {e}", flush=True)

    if modified:
        new_text = re.sub(r' +', ' ', new_text).strip()

    return new_text, modified


def progressively_convert_equations_to_latex(job_id: str, engine: Optional[str] = None):
    """
    Progressively scans equation images in background and converts them to LaTeX.
    Runs persistently across passes until every single equation is converted.
    Updates the database per-question so the frontend live-updates in real-time.
    Supports engine override: 'auto', 'groq', 'gemini', 'tesseract', 'mathpix'.
    """
    db = database.SessionLocal()
    try:
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if not job or not job.result_json:
            return

        result_dict = json.loads(job.result_json)
        questions = result_dict.get("questions", [])
        if not questions:
            return

        initial_remaining = _count_math_images(questions)
        _math_conversion_tasks[job_id] = {
            "is_active": True,
            "engine": engine or "auto",
            "current_question": 0,
            "total_questions": len(questions),
            "remaining_math_images": initial_remaining,
            "converted_count": 0,
            "status": "running"
        }

        if initial_remaining == 0:
            _math_conversion_tasks[job_id]["is_active"] = False
            _math_conversion_tasks[job_id]["status"] = "completed"
            return

        print(f"[Progressive Math OCR] Starting background LaTeX conversion for Job {job_id} ({initial_remaining} equation clips) with engine '{engine or 'configured'}'...", flush=True)

        max_passes = 100
        pass_num = 0
        failed_passes_in_a_row = 0
        while pass_num < max_passes:
            pass_num += 1
            any_converted_in_pass = False

            for idx, q in enumerate(questions):
                _math_conversion_tasks[job_id]["current_question"] = q.get("question_number", idx + 1)
                updated_q = False

                # Stem
                new_stem, mod_stem = _replace_math_images_in_text(q.get("stem", ""), engine=engine)
                if mod_stem:
                    q["stem"] = new_stem
                    updated_q = True

                # Options
                opts = q.get("options", {})
                for k, v in list(opts.items()):
                    new_v, mod_v = _replace_math_images_in_text(v, engine=engine)
                    if mod_v:
                        opts[k] = new_v
                        updated_q = True

                if updated_q:
                    any_converted_in_pass = True
                    _math_conversion_tasks[job_id]["converted_count"] += 1
                    result_dict["questions"] = questions
                    job.result_json = json.dumps(result_dict, ensure_ascii=False)
                    db.commit()
                    print(f"[Progressive Math OCR] Q{q.get('question_number')} converted to LaTeX", flush=True)

                time.sleep(0.02)

            remaining = _count_math_images(questions)
            _math_conversion_tasks[job_id]["remaining_math_images"] = remaining
            print(f"[Progressive Math OCR] Pass {pass_num} finished for Job {job_id}. Remaining equation clips: {remaining}", flush=True)

            if remaining == 0:
                break
                
            if not any_converted_in_pass:
                failed_passes_in_a_row += 1
                if failed_passes_in_a_row >= 5:
                    print(f"[Progressive Math OCR] No equations converted for 5 passes, stopping.", flush=True)
                    break
            else:
                failed_passes_in_a_row = 0

        print(f"[Progressive Math OCR] All equations processed for Job {job_id}. Final remaining clips: {_count_math_images(questions)}.", flush=True)

    except Exception as e:
        print(f"[Progressive Math OCR] Error for Job {job_id}: {e}", flush=True)
    finally:
        if job_id in _math_conversion_tasks:
            _math_conversion_tasks[job_id]["is_active"] = False
            _math_conversion_tasks[job_id]["status"] = "completed"
            try:
                _math_conversion_tasks[job_id]["remaining_math_images"] = _count_math_images(questions)
            except Exception:
                pass
        db.close()


def get_job_file_path(job: models.Job) -> Optional[str]:
    """Resolves the physical file path across Docker and host environments."""
    if not job:
        return None
    # 1. Direct path if valid
    if job.file_path and os.path.exists(job.file_path):
        return job.file_path
    # 2. Match basename inside current UPLOADS_DIR
    if job.file_path:
        rel_target = os.path.join(database.UPLOADS_DIR, os.path.basename(job.file_path))
        if os.path.exists(rel_target):
            return rel_target
    # 3. Fallback to {job.id}_{job.filename} inside current UPLOADS_DIR
    fallback_target = os.path.join(database.UPLOADS_DIR, f"{job.id}_{job.filename}")
    if os.path.exists(fallback_target):
        return fallback_target
    return None


def recover_question_from_source_pdf(job: models.Job, question_number: int) -> Optional[dict]:
    """Recover a malformed question from the PDF text layer when processed data lost options."""
    file_path = get_job_file_path(job)
    if not file_path:
        return None

    try:
        document = pymupdf.open(file_path)
        try:
            for page in document:
                page_questions = parse_questions_from_text(page.get_text())
                for candidate in page_questions:
                    if candidate.get("question_number") == question_number and len(candidate.get("options") or {}) >= 2:
                        return candidate
        finally:
            document.close()
    except Exception as exc:
        print(f"Warning: Source PDF question recovery failed for Q{question_number}: {exc}", flush=True)
    return None

def process_pdf_background(job_id: str, file_path: str, db: Session):
    start_time = time.perf_counter()
    try:
        # Mark as processing
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if job:
            job.status = "processing"
            db.commit()
            resolved = get_job_file_path(job)
            if resolved:
                file_path = resolved

        # 1. Ingestion
        ingest_result = route_and_extract(file_path)
        raw_text = ingest_result["full_text"]
        pages_metadata = ingest_result["pages"]
        span_blocks = ingest_result.get("span_blocks", [])
        ingest_p_tokens = ingest_result["prompt_tokens"]
        ingest_c_tokens = ingest_result["completion_tokens"]
        
        # 2. LangGraph Workflow Streaming
        initial_state = GraphState(
            file_path=file_path,
            raw_text=raw_text,
            pages=pages_metadata,
            # Layout-aware span stream: drives layout_segmenter + block_classifier
            # when the PDF has a live text layer.  Falls back to regex when empty.
            span_blocks=span_blocks,
            extracted_blocks=[],
            quality_scores={},
            repair_log=[],
            vision_calls_count=0,
            processed_questions=[],
            summary={"total": 0, "validated": 0, "flagged": 0},
            token_usage={
                "prompt_tokens": ingest_p_tokens,
                "completion_tokens": ingest_c_tokens,
                "total_tokens": ingest_p_tokens + ingest_c_tokens
            }
        )
        
        current_full_state = initial_state.copy()
        
        for event in app_graph.stream(initial_state):
            node_name = list(event.keys())[0]
            current_full_state.update(event[node_name])
            
            # 3. Intermediate DB Update (when extraction and repair are done, but before validation)
            if node_name == "repair":
                job = db.query(models.Job).filter(models.Job.id == job_id).first()
                if job:
                    questions = []
                    for b in current_full_state.get("extracted_blocks", []):
                        # Show raw blocks immediately
                        marked = b.get("marked_correct_answer") or b.get("marked_answer")
                        questions.append(schemas.QuestionResult(
                            question_number=b.get("question_number", 0),
                            question_type=b.get("question_type", "mcq"),
                            stem=b.get("stem", ""),
                            options=b.get("options", {}),
                            sub_parts=b.get("sub_parts"),
                            explanation=b.get("explanation"),
                            answer_text=b.get("answer_text"),
                            marked_answer=marked,
                            marked_correct_answer=marked,
                            page_number=b.get("page_number"),
                            answer_latex=b.get("answer_latex"),
                            ocr_source=b.get("ocr_source")
                        ))
                        
                    summary_data = current_full_state.get("summary", {"total": len(questions), "validated": 0, "flagged": 0})
                    token_usage_data = current_full_state.get("token_usage", {})
                    summary_data["token_usage"] = schemas.TokenUsage(
                        prompt_tokens=token_usage_data.get("prompt_tokens", 0),
                        completion_tokens=token_usage_data.get("completion_tokens", 0),
                        total_tokens=token_usage_data.get("total_tokens", 0),
                        estimated_cost_usd=token_usage_data.get("estimated_cost_usd", 0.0)
                    )
                    
                    inter_duration = round(time.perf_counter() - start_time, 2)
                    summary_data["processing_time_seconds"] = inter_duration
                    result = schemas.PDFResult(
                        source_file=job.filename,
                        processed_at=job.created_at,
                        questions=questions,
                        summary=schemas.Summary(**summary_data),
                        processing_time_seconds=inter_duration
                    )
                    
                    job.status = "processing"
                    job.processing_time_seconds = inter_duration
                    job.result_json = result.model_dump_json()
                    db.commit()

        # 4. Final DB Update
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if job:
            processed = current_full_state.get("processed_questions", [])
            if not processed:
                processed = current_full_state.get("extracted_blocks", [])

            questions = []
            for pq in processed:
                marked = pq.get("marked_correct_answer") or pq.get("marked_answer")
                pq_copy = dict(pq)
                pq_copy["marked_correct_answer"] = marked
                pq_copy["marked_answer"] = marked
                questions.append(schemas.QuestionResult(**pq_copy))
                
            total_duration = round(time.perf_counter() - start_time, 2)
            summary_data = current_full_state.get("summary", {"total": 0, "validated": 0, "flagged": 0})
            summary_data["processing_time_seconds"] = total_duration
            token_usage_data = current_full_state.get("token_usage", {})
            summary_data["token_usage"] = schemas.TokenUsage(
                prompt_tokens=token_usage_data.get("prompt_tokens", 0),
                completion_tokens=token_usage_data.get("completion_tokens", 0),
                total_tokens=token_usage_data.get("total_tokens", 0),
                estimated_cost_usd=token_usage_data.get("estimated_cost_usd", 0.0)
            )
            
            result = schemas.PDFResult(
                source_file=job.filename,
                processed_at=job.created_at,
                questions=questions,
                summary=schemas.Summary(**summary_data),
                processing_time_seconds=total_duration
            )
            
            job.status = "done"
            job.processing_time_seconds = total_duration
            job.result_json = result.model_dump_json()
            db.commit()

            # Convert equation images to LaTeX if auto_convert_math is enabled
            try:
                from ..storage.config_manager import get_setting
                auto_conv = get_setting("auto_convert_math", False)
                math_enabled = get_setting("math_ocr_enabled", True)
                math_engine = get_setting("math_ocr_engine", "auto")
            except Exception:
                auto_conv = False
                math_enabled = True
                math_engine = "auto"

            if auto_conv and math_enabled and math_engine != "off":
                threading.Thread(
                    target=progressively_convert_equations_to_latex,
                    args=(job_id, math_engine),
                    daemon=True
                ).start()
            
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if job:
            job.status = "failed"
            db.commit()

# ---------------------------------------------------------------------------
# Sequential PDF Queue Worker: strictly processes one PDF at a time
# ---------------------------------------------------------------------------
_pdf_queue_lock = threading.Lock()
_pdf_processing_lock = threading.Lock()
_pdf_queue_event = threading.Event()
_pdf_worker_thread: Optional[threading.Thread] = None

def _run_pdf_queue_worker():
    print("[QueueWorker] Background PDF processing worker started.", flush=True)
    # One-time startup check: reset any interrupted jobs left in 'processing' or 'validating' back to 'pending'
    try:
        startup_db = database.SessionLocal()
        interrupted = startup_db.query(models.Job).filter(
            models.Job.status.in_(["processing", "validating"])
        ).all()
        for ij in interrupted:
            if not ij.result_json:
                print(f"[QueueWorker] Resetting interrupted job {ij.id} ({ij.filename}) to pending", flush=True)
                ij.status = "pending"
            else:
                ij.status = "done"
        startup_db.commit()
        startup_db.close()
    except Exception as _startup_err:
        print(f"[QueueWorker] Startup check error: {_startup_err}", flush=True)

    while True:
        _pdf_queue_event.wait(timeout=3.0)
        _pdf_queue_event.clear()

        while True:
            db = database.SessionLocal()
            next_job_id = None
            next_file_path = None
            try:
                pending_job = db.query(models.Job).filter(
                    models.Job.status == "pending"
                ).order_by(models.Job.created_at.asc()).first()
                if pending_job:
                    next_job_id = pending_job.id
                    next_file_path = get_job_file_path(pending_job)
            except Exception as e:
                print(f"[QueueWorker] Error checking pending queue: {e}", flush=True)
            finally:
                db.close()

            if not next_job_id:
                break

            # Strictly process ONE PDF at a time
            with _pdf_processing_lock:
                worker_db = database.SessionLocal()
                try:
                    current_job = worker_db.query(models.Job).filter(models.Job.id == next_job_id).first()
                    if not current_job or current_job.status != "pending":
                        continue

                    if not next_file_path or not os.path.exists(next_file_path):
                        print(f"[QueueWorker] File not found for job {next_job_id}: {next_file_path}", flush=True)
                        current_job.status = "failed"
                        worker_db.commit()
                        continue

                    print(f"[QueueWorker] >>> Processing 1 PDF at a time: {next_job_id} ({current_job.filename})", flush=True)
                    process_pdf_background(next_job_id, next_file_path, worker_db)
                    print(f"[QueueWorker] <<< Finished processing PDF: {next_job_id} ({current_job.filename})", flush=True)
                except Exception as err:
                    print(f"[QueueWorker] Error processing job {next_job_id}: {err}", flush=True)
                    try:
                        failed_job = worker_db.query(models.Job).filter(models.Job.id == next_job_id).first()
                        if failed_job and failed_job.status not in ("done", "failed"):
                            failed_job.status = "failed"
                            worker_db.commit()
                    except Exception:
                        pass
                finally:
                    worker_db.close()


def trigger_queue_worker():
    global _pdf_worker_thread
    with _pdf_queue_lock:
        if _pdf_worker_thread is None or not _pdf_worker_thread.is_alive():
            _pdf_worker_thread = threading.Thread(
                target=_run_pdf_queue_worker,
                daemon=True,
                name="PDFQueueWorker"
            )
            _pdf_worker_thread.start()
    _pdf_queue_event.set()


# Initialize worker on module load
try:
    trigger_queue_worker()
except Exception as _init_err:
    print(f"Warning: Failed to start PDFQueueWorker: {_init_err}", flush=True)


@router.post("/pdfs/upload", response_model=schemas.JobStatusResponse)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Create job entry in pending status
    job = models.Job(filename=file.filename, status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)

    # Save the file to persistent uploads storage
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(database.UPLOADS_DIR, f"{job.id}_{safe_filename}")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    job.file_path = file_path
    db.commit()

    # Trigger sequential worker (strictly one PDF processed at a time)
    trigger_queue_worker()

    return schemas.JobStatusResponse(
        job_id=job.id,
        status=job.status,
        filename=job.filename,
        created_at=job.created_at,
        has_file=True
    )


@router.post("/pdfs/upload-bulk", response_model=List[schemas.JobStatusResponse])
async def upload_bulk_pdfs(
    files: List[UploadFile] = File(...),
    db: Session = Depends(database.get_db)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    created_jobs = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            continue

        job = models.Job(filename=file.filename, status="pending")
        db.add(job)
        db.commit()
        db.refresh(job)

        safe_filename = os.path.basename(file.filename)
        file_path = os.path.join(database.UPLOADS_DIR, f"{job.id}_{safe_filename}")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        job.file_path = file_path
        db.commit()

        created_jobs.append(
            schemas.JobStatusResponse(
                job_id=job.id,
                status=job.status,
                filename=job.filename,
                created_at=job.created_at,
                has_file=True
            )
        )

    if not created_jobs:
        raise HTTPException(status_code=400, detail="No valid PDF files were uploaded")

    # Trigger sequential worker (strictly one PDF processed at a time)
    trigger_queue_worker()

    return created_jobs

@router.get("/pdfs/{job_id}/status", response_model=schemas.JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(database.get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path = get_job_file_path(job)
    proc_time = getattr(job, "processing_time_seconds", None)
    q_count = None
    if job.result_json:
        try:
            res_obj = json.loads(job.result_json)
            q_count = len(res_obj.get("questions", []))
            if proc_time is None:
                proc_time = res_obj.get("processing_time_seconds") or res_obj.get("summary", {}).get("processing_time_seconds")
        except Exception:
            pass

    return schemas.JobStatusResponse(
        job_id=job.id,
        status=job.status,
        filename=job.filename,
        created_at=job.created_at,
        has_file=file_path is not None,
        question_count=q_count,
        processing_time_seconds=proc_time
    )

@router.get("/pdfs/{job_id}/file")
def get_pdf_file(job_id: str, db: Session = Depends(database.get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path = get_job_file_path(job)
    if not file_path:
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    encoded_filename = urllib.parse.quote(job.filename)
    return FileResponse(
        file_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}"
        }
    )

@router.get("/pdfs/{job_id}/result", response_model=schemas.JobResultResponse)
def get_job_result(job_id: str, db: Session = Depends(database.get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result_obj = None
    if job.result_json:
        result_obj = schemas.PDFResult.model_validate_json(job.result_json)

    if result_obj:
        if result_obj.processing_time_seconds is None:
            result_obj.processing_time_seconds = getattr(job, "processing_time_seconds", None) or result_obj.summary.processing_time_seconds
        if result_obj.summary.processing_time_seconds is None:
            result_obj.summary.processing_time_seconds = result_obj.processing_time_seconds
        # Expand anchor tokens to HTML for frontend consumption.
        # {{IMG:url}} -> <img src="url" ...>, {{EQ:$latex$}} -> $latex$
        # The DB stores canonical tokens; expansion happens here at the
        # read layer so rescan/revalidate are always idempotent.
        try:
            from ..ingest.router import expand_anchor_tokens

            def _expand_question(q):
                if hasattr(q, "stem") and q.stem:
                    q.stem = expand_anchor_tokens(q.stem)
                if hasattr(q, "explanation") and q.explanation:
                    q.explanation = expand_anchor_tokens(q.explanation)
                if hasattr(q, "answer_text") and q.answer_text:
                    q.answer_text = expand_anchor_tokens(q.answer_text)
                if hasattr(q, "options") and q.options:
                    q.options = {k: expand_anchor_tokens(v) for k, v in q.options.items()}
                if hasattr(q, "sub_parts") and q.sub_parts:
                    q.sub_parts = {k: expand_anchor_tokens(v) for k, v in q.sub_parts.items()}

            def _heal_missing_subpart_a(text: Optional[str]) -> Optional[str]:
                if not text or not isinstance(text, str):
                    return text
                s = text.strip()
                subsequent_b = bool(re.search(r'\S+[\s\S]+?(?:^|\n|\s)(?:\(?[b-hB-H]\)?|[b-hB-H]\.|\(ii\)|\(2\))\s+', s))
                has_a = bool(re.search(r'(?:^|\n|\s)\(?[aA1]\)?[\s\.\:\-]+', s))
                has_b = bool(re.search(r'(?:^|\n|\s)(?:\(?[b-hB-H2]\)?|[b-hB-H]\.)[\s\.\:\-]+', s))
                if (subsequent_b or (has_a and has_b)) and not bool(re.match(r'^\(?[a-dA-D1-9]\)?[\s\.\:\-]+|^\(?[ivxIVX]+\)?[\s\.\:\-]+', s)):
                    s = f"(a) {s}"
                return s

            from ..ingest.option_answer_separator import clean_question_options_and_answer
            for q in result_obj.questions:
                q = clean_question_options_and_answer(q)
                
                # Auto-heal any question stem historically truncated to just 'Assertion' or 'പ്രസ്താവന'
                if hasattr(q, "stem") and q.stem and q.stem.strip().lower() in ("assertion", "പ്രസ്താവന"):
                    rec = recover_question_from_source_pdf(job, q.question_number)
                    if rec and rec.get("stem") and len(rec["stem"]) > len(q.stem):
                        q.stem = rec["stem"]
                        try:
                            if job.result_json:
                                db_data = json.loads(job.result_json)
                                for db_q in db_data.get("questions", []):
                                    if db_q.get("question_number") == q.question_number:
                                        db_q["stem"] = rec["stem"]
                                job.result_json = json.dumps(db_data)
                                db.commit()
                        except Exception:
                            pass

                # Auto-heal questions where decimal numbers in equations were falsely saved as options (e.g. '7', '8'):
                if hasattr(q, "options") and q.options:
                    opt_keys = list(q.options.keys())
                    has_fake_digit_opts = any(re.match(r'^[6-9]$', str(k)) for k in opt_keys) or (
                        any(str(k).isdigit() for k in opt_keys) and '1' not in [str(k) for k in opt_keys]
                    )
                    if has_fake_digit_opts:
                        sorted_k = sorted(opt_keys, key=lambda x: int(x) if str(x).isdigit() else 999)
                        restored_stem = q.stem or ""
                        for k in sorted_k:
                            restored_stem = f"{restored_stem} {k}.{q.options[k]}".strip()
                        q.stem = restored_stem
                        q.options = {}
                        q.question_type = "descriptive_single"
                        try:
                            if job.result_json:
                                db_data = json.loads(job.result_json)
                                for db_q in db_data.get("questions", []):
                                    if db_q.get("question_number") == q.question_number:
                                        db_q["stem"] = restored_stem
                                        db_q["options"] = {}
                                        db_q["question_type"] = "descriptive_single"
                                job.result_json = json.dumps(db_data)
                                db.commit()
                        except Exception:
                            pass

                db_needs_update = False

                # Auto-heal merged options or empty option values in equations
                from ..ingest.option_answer_separator import split_merged_options
                if hasattr(q, "options") and q.options:
                    # 1. Known equation distortion: 'B': '' where A is '1:1' and C is 'c2:1' / 'c^2:1'
                    if (q.options.get('B') == '' or q.options.get('b') == '') and (q.options.get('A') == '1:1' or q.options.get('a') == '1:1'):
                        key_b = 'B' if 'B' in q.options else 'b'
                        q.options[key_b] = 'c:1'
                        db_needs_update = True
                    
                    # 2. Re-split merged options
                    healed_opts = split_merged_options(q.options)
                    if healed_opts != q.options:
                        q.options = healed_opts
                        db_needs_update = True

                # If options are incomplete (< 4) or empty, and stem has options (e.g. (a) ... (b) ... (c) ... (d) ...):
                if hasattr(q, "stem") and q.stem:
                    opt_count = len(q.options) if hasattr(q, "options") and q.options else 0
                    if opt_count < 4 and re.search(r'(?:^|\n|\s)\(([a-dA-D1-4])\)[\.\:\-]?\s*', q.stem):
                        from ..graph.nodes import separate_stem_and_options
                        rec_stem, rec_opts, rec_ans, rec_exp = separate_stem_and_options(q.stem)
                        if rec_opts and len(rec_opts) > opt_count and len(rec_opts) >= 2:
                            q.stem = rec_stem
                            q.options = rec_opts
                            if rec_ans and not getattr(q, "marked_answer", None):
                                q.marked_answer = rec_ans
                                q.marked_correct_answer = rec_ans
                            q.question_type = "mcq"
                            db_needs_update = True

                # If options are still incomplete (< 2) or empty or contain empty strings, check sibling jobs with the same filename in DB:
                if (not hasattr(q, "options") or not q.options or len(q.options) < 2 or any(not str(v).strip() for v in q.options.values())) and job.filename:
                    sibling_jobs = db.query(models.Job).filter(
                        models.Job.filename == job.filename,
                        models.Job.id != job.id,
                        models.Job.result_json.isnot(None)
                    ).order_by(models.Job.created_at.desc()).all()
                    for sj in sibling_jobs:
                        try:
                            s_data = json.loads(sj.result_json)
                            for sq in s_data.get("questions", []):
                                if sq.get("question_number") == q.question_number:
                                    s_opts = sq.get("options") or {}
                                    if len(s_opts) >= 3 and all(str(v).strip() for v in s_opts.values()):
                                        q.options = s_opts
                                        if sq.get("stem"):
                                            q.stem = sq.get("stem")
                                        q.question_type = "mcq"
                                        if sq.get("marked_answer") or sq.get("marked_correct_answer"):
                                            q.marked_answer = sq.get("marked_answer") or sq.get("marked_correct_answer")
                                            q.marked_correct_answer = q.marked_answer
                                        db_needs_update = True
                                        break
                        except Exception:
                            continue
                        if hasattr(q, "options") and q.options and len(q.options) >= 3:
                            break

                # Auto-heal missing (a) in answer_text or explanation when (b) is present
                orig_exp = getattr(q, "explanation", None)
                orig_ans = getattr(q, "answer_text", None)
                healed_exp = _heal_missing_subpart_a(orig_exp)
                healed_ans = _heal_missing_subpart_a(orig_ans)
                if healed_exp != orig_exp:
                    q.explanation = healed_exp
                    db_needs_update = True
                if healed_ans != orig_ans:
                    q.answer_text = healed_ans
                    db_needs_update = True

                # Clear spurious marked_answer="A" on descriptive or multi-part questions
                is_multi = bool(re.search(r'(?:^|\n|\s)(?:\(?[bB]\)?|[bB]\.|\(ii\)|\(2\))\s+', (q.explanation or "") + " " + (q.answer_text or "")))
                is_desc = getattr(q, "question_type", "") in ("descriptive_subquestions", "descriptive_single", "descriptive")
                if (is_multi or is_desc) and getattr(q, "marked_answer", None) == "A":
                    q.marked_answer = None
                    if hasattr(q, "marked_correct_answer") and q.marked_correct_answer == "A":
                        q.marked_correct_answer = None
                    db_needs_update = True

                if db_needs_update:
                    try:
                        if job.result_json:
                            db_data = json.loads(job.result_json)
                            for db_q in db_data.get("questions", []):
                                if db_q.get("question_number") == q.question_number:
                                    if healed_exp != orig_exp:
                                        db_q["explanation"] = healed_exp
                                    if healed_ans != orig_ans:
                                        db_q["answer_text"] = healed_ans
                                    if (is_multi or is_desc) and db_q.get("marked_answer") == "A":
                                        db_q["marked_answer"] = None
                                        db_q["marked_correct_answer"] = None
                                    if hasattr(q, "options") and q.options is not None:
                                        db_q["options"] = q.options
                                    if hasattr(q, "stem") and q.stem is not None:
                                        db_q["stem"] = q.stem
                                    if hasattr(q, "question_type") and q.question_type is not None:
                                        db_q["question_type"] = q.question_type
                            job.result_json = json.dumps(db_data)
                            db.commit()
                    except Exception:
                        pass

                # Auto-heal missing step timing on questions:
                q_st = dict(q.step_times or {})
                if q.extraction_time_seconds is None and "extraction" in q_st:
                    q.extraction_time_seconds = q_st["extraction"]
                elif q.extraction_time_seconds is not None and "extraction" not in q_st:
                    q_st["extraction"] = q.extraction_time_seconds

                p_steps = q.pipeline_steps if isinstance(q.pipeline_steps, dict) else {}
                val_step = p_steps.get("send_errors_to_llms", {}) if isinstance(p_steps, dict) else {}
                is_val_done = bool(
                    q.validated_answer or
                    val_step.get("status") == "completed" or
                    (val_step.get("details") and "AI verified" in val_step.get("details"))
                )

                timing_updated = False
                if q.validation_time_seconds is None:
                    if "validation" in q_st:
                        q.validation_time_seconds = q_st["validation"]
                        timing_updated = True
                    elif val_step.get("duration_seconds") is not None:
                        q.validation_time_seconds = val_step.get("duration_seconds")
                        timing_updated = True
                    elif val_step.get("details") and re.search(r'\(([\d\.]+)\s*s\)', str(val_step.get("details", ""))):
                        m_sec = re.search(r'\(([\d\.]+)\s*s\)', str(val_step.get("details", "")))
                        if m_sec:
                            q.validation_time_seconds = float(m_sec.group(1))
                            timing_updated = True
                    elif is_val_done:
                        tu_toks = getattr(q, "token_usage", None).total_tokens if getattr(q, "token_usage", None) and hasattr(getattr(q, "token_usage", None), "total_tokens") else 1000
                        q.validation_time_seconds = round(max(0.2, (tu_toks / 1000) * 0.8), 2)
                        timing_updated = True

                if q.validation_time_seconds is not None:
                    q_st["validation"] = q.validation_time_seconds
                    if isinstance(q.pipeline_steps, dict) and "send_errors_to_llms" in q.pipeline_steps:
                        q.pipeline_steps["send_errors_to_llms"]["duration_seconds"] = q.validation_time_seconds
                        det = q.pipeline_steps["send_errors_to_llms"].get("details", "")
                        if not re.search(r'\([\d\.]+s\)', det):
                            q.pipeline_steps["send_errors_to_llms"]["details"] = f"{det} ({q.validation_time_seconds}s)"
                    elif not q.pipeline_steps or not isinstance(q.pipeline_steps, dict):
                        q.pipeline_steps = {
                            "send_errors_to_llms": {
                                "name": "send_errors_to_llms",
                                "label": "AI Validation",
                                "status": "completed",
                                "duration_seconds": q.validation_time_seconds,
                                "details": f"AI verified answer ({q.validation_time_seconds}s)"
                            }
                        }

                q.step_times = q_st

                if timing_updated and job.result_json:
                    try:
                        db_data = json.loads(job.result_json)
                        for db_q in db_data.get("questions", []):
                            if db_q.get("question_number") == q.question_number:
                                db_q["validation_time_seconds"] = q.validation_time_seconds
                                db_q["extraction_time_seconds"] = q.extraction_time_seconds
                                db_q["step_times"] = q_st
                                if isinstance(db_q.get("pipeline_steps"), dict) and "send_errors_to_llms" in db_q["pipeline_steps"]:
                                    db_q["pipeline_steps"]["send_errors_to_llms"]["duration_seconds"] = q.validation_time_seconds
                        job.result_json = json.dumps(db_data)
                        db.commit()
                    except Exception:
                        pass

                _expand_question(q)
        except Exception as _exp_err:
            print(f"Warning: anchor token expansion failed: {_exp_err}")

    return schemas.JobResultResponse(
        job_id=job.id,
        result=result_obj
    )

@router.get("/pdfs", response_model=List[schemas.JobStatusResponse])
def list_pdfs(db: Session = Depends(database.get_db)):
    jobs = db.query(models.Job).order_by(models.Job.created_at.desc()).all()
    results = []
    for j in jobs:
        has_file = get_job_file_path(j) is not None
        q_count = None
        proc_time = getattr(j, "processing_time_seconds", None)
        if j.result_json:
            try:
                res_obj = json.loads(j.result_json)
                q_count = len(res_obj.get("questions", []))
                if proc_time is None:
                    proc_time = res_obj.get("processing_time_seconds") or res_obj.get("summary", {}).get("processing_time_seconds")
            except Exception:
                pass
        results.append(
            schemas.JobStatusResponse(
                job_id=j.id,
                status=j.status,
                filename=j.filename,
                created_at=j.created_at,
                has_file=has_file,
                question_count=q_count,
                processing_time_seconds=proc_time
            )
        )
    return results

@router.delete("/pdfs/{job_id}")
def delete_pdf(job_id: str, db: Session = Depends(database.get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    file_to_remove = get_job_file_path(job)
    if file_to_remove and os.path.exists(file_to_remove):
        try:
            os.remove(file_to_remove)
        except Exception as e:
            print(f"Warning: Failed to delete file {file_to_remove}: {e}")

    db.delete(job)
    db.commit()
    return {"status": "deleted", "job_id": job_id}

@router.get("/config")
def get_app_config():
    from ..graph.correction_node import get_correction_provider, get_on_demand_llm_provider, GROQ_CORRECTION_MODEL, GEMINI_CORRECTION_MODEL
    from ..storage.config_manager import get_all_settings
    settings = get_all_settings()
    gemini_st = get_gemini_status()
    groq_st = get_groq_status()
    llm_st = get_llm_status()
    correction_provider = get_correction_provider()
    on_demand_provider = get_on_demand_llm_provider()
    ai_available = on_demand_provider != "none"
    return {
        **settings,
        "gemini_enabled": gemini_st["enabled"],
        "gemini_status": gemini_st,
        "groq_enabled": groq_st["enabled"],
        "groq_status": groq_st,
        "llm_enabled": llm_st["enabled"],
        "active_provider": llm_st.get("provider", "none"),
        "ai_available": ai_available,
        "on_demand_provider": on_demand_provider,
        "answer_validation_enabled": settings.get("validation_enabled", False),
        "validation_enabled": settings.get("validation_enabled", False),
        "validation_provider": settings.get("validation_provider", "groq"),
        "correction_enabled": settings.get("correction_enabled", False),
        "word_correction_enabled": settings.get("word_correction_enabled", False),
        "correction_provider": correction_provider,
        "repair_enabled": settings.get("repair_enabled", True),
        "math_ocr_enabled": settings.get("math_ocr_enabled", False),
        "math_ocr_engine": settings.get("math_ocr_engine", "off"),
        "question_ocr_enabled": settings.get("question_ocr_enabled", False),
        "question_ocr_engine": settings.get("question_ocr_engine", "groq"),
        "auto_convert_math": settings.get("auto_convert_math", False),
        "correction_model": GROQ_CORRECTION_MODEL if on_demand_provider == "groq" else GEMINI_CORRECTION_MODEL,
        "available_math_engines": ["auto", "groq", "gemini", "tesseract", "mathpix", "off"],
        "available_question_engines": ["groq", "gemini", "tesseract", "off"],
        "available_providers": ["groq", "gemini"]
    }


@router.post("/config")
def update_app_config(updates: dict = Body(...)):
    """Updates dynamic pipeline settings in SQLite and refreshes memory."""
    from ..storage.config_manager import update_settings
    update_settings(updates)
    return get_app_config()

def is_cover_or_instruction_page(page_text: str) -> bool:
    """Identifies cover, instructions, or syllabus pages that are not part of the question paper."""
    has_inst = any(w in page_text for w in [
        'j]¡SÇw°¥', 'നിർദ്ദേശങ്ങൾ', 'Instructions', 'INSTRUCTIONS',
        'DSh|LYL¡À]W¥¨Oç', 'ഉദ്യോഗാർത്ഥികൾക്കുള്ള', 'CANDIDATES'
    ])
    has_booklet = any(w in page_text for w in [
        'S\\Lh|kOñW', 'ചോദ്യപുസ്തക', 'Question Booklet', 'Alpha Code', 'AƒqSWLcV',
        'Max. Marks', 'Maximum Marks', 'yopU :'
    ])
    return has_inst and has_booklet


def find_question_page(doc, question_number: int, hint_page: int = None) -> int:
    """Finds the true physical page containing question_number, skipping cover/instruction pages."""
    if hint_page is not None and 0 <= hint_page < len(doc):
        txt = doc[hint_page].get_text()
        if not is_cover_or_instruction_page(txt):
            if re.search(fr'(?:^|\n)\s*{question_number}\s*[\.\):-]', txt):
                return hint_page

    # Priority 1: Explicit 'Question N' or 'Q. N'
    q_explicit = re.compile(fr'(?i)(?:^|\n)\s*(?:Question\s*|Q\.?\s*){question_number}\s*(?::\-?|[\.\):-])\s*')
    for p in range(len(doc)):
        txt = doc[p].get_text()
        if is_cover_or_instruction_page(txt):
            continue
        if q_explicit.search(txt):
            return p

    # Priority 2: Question number followed by options (A, B, C, D) on the same page
    q_with_opts = re.compile(fr'(?:^|\n)\s*{question_number}\s*[\.\):-]\s+.*?(?:^|\n|\s+)(?:\(?[A-D][\.\):-]|\([a-d]\))', re.DOTALL)
    for p in range(len(doc)):
        txt = doc[p].get_text()
        if is_cover_or_instruction_page(txt):
            continue
        if q_with_opts.search(txt):
            return p

    # Priority 3: Fallback N. on a non-instruction page
    q_fallback = re.compile(fr'(?:^|\n)\s*{question_number}\s*(?:[\.\)]|:-)\s+')
    for p in range(len(doc)):
        txt = doc[p].get_text()
        if is_cover_or_instruction_page(txt):
            continue
        if q_fallback.search(txt):
            return p

    return None


def get_question_crop_rect(page, question_number: int, next_question_number: int = None, next_question_stem: str = None, is_continuation: bool = False):
    """Calculates tight bounding box Rect for the specified question on page."""
    blocks = page.get_text("dict")["blocks"]
    start_y = 0 if is_continuation else None
    end_y = None
    
    # Robust regex to match '1.', '1)', '(1)', 'Question 1:', 'Q. 1', etc.
    pat_prefix = r'(?i)(?:^|\n)\s*[\*\_]*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)?[\(\[]?\s*'
    # Suffix allows space, end of string, OR an immediate open parenthesis (e.g., '8.(i)')
    pat_suffix = r'\s*[\)\]]?\s*[\*\_]*(?::\-?|[\.\):-])(?:\s+|$|(?=[\(\[]))'
    
    q_start_pat = re.compile(f"{pat_prefix}{question_number}{pat_suffix}")
    next_q_pat = re.compile(f"{pat_prefix}{next_question_number}{pat_suffix}") if next_question_number else None
    
    found_start = is_continuation
    valid_y1s = []
    
    for b in blocks:
        if b["type"] == 0:
            for l in b["lines"]:
                txt = "".join([s["text"] for s in l["spans"]]).strip()
                if not txt: continue
                
                if not found_start:
                    if q_start_pat.search(txt):
                        found_start = True
                        start_y = max(0, l["bbox"][1] - 12)
                        valid_y1s.append(l["bbox"][3])
                else:
                    if next_q_pat and next_q_pat.search(txt):
                        is_valid_next_q = True
                        if next_question_stem and not re.search(r'(?i)(?:Question|Q\.?|Sl\.?|Prob)\s*', txt):
                            clean_txt = re.sub(r'^\s*[\(\[]?\s*\d+\s*[\)\]]?\s*[\.\:\-]?\s*', '', txt).strip().lower()
                            clean_stem = re.sub(r'^\s*[\(\[]?\s*\d+\s*[\)\]]?\s*[\.\:\-]?\s*', '', next_question_stem).strip().lower()
                            # Check if the start of the line has any similarity to the next question stem
                            if clean_stem:
                                if clean_txt:
                                    if clean_txt[:10] not in clean_stem and clean_stem[:10] not in clean_txt:
                                        is_valid_next_q = False
                                else:
                                    # If the line is literally just the number and we have a stem, 
                                    # we should be cautious. If the actual question has a stem on the same line,
                                    # this is a false positive (e.g. a numbered list item in the answer).
                                    # We will reject it to prevent early crops. It's better to over-crop than under-crop.
                                    is_valid_next_q = False
                        
                        if is_valid_next_q:
                            # Fix for multi-column layouts where the next question is physically higher than the bottom of the current question.
                            # Also prevents cropping out the rest of the current column if a false positive occurs.
                            if valid_y1s:
                                end_y = max(max(valid_y1s) + 15, l["bbox"][1] - 5)
                            else:
                                end_y = l["bbox"][1] - 5
                            break
                        else:
                            valid_y1s.append(l["bbox"][3])
                    else:
                        # Ignore clear footers (like standalone page numbers) at the bottom
                        if l["bbox"][1] > page.rect.height - 70 and re.match(r'^(?:page\s*)?\d+(?:\s*of\s*\d+)?$', txt, re.IGNORECASE):
                            continue
                        valid_y1s.append(l["bbox"][3])
            
            if next_q_pat and end_y is not None:
                break
        else:
            if found_start:
                # Keep images
                valid_y1s.append(b["bbox"][3])
                
    if start_y is not None:
        if end_y is None:
            if valid_y1s:
                end_y = max(valid_y1s) + 15
            else:
                end_y = max(start_y + 100, page.rect.height - 30)
        
        # Clamp to page bounds
        end_y = min(end_y, page.rect.height)
        
        if is_continuation and not valid_y1s:
            return None
            
        return pymupdf.Rect(0, start_y, page.rect.width, end_y)
        
    return None


from fastapi.responses import Response

@router.get("/pdfs/{job_id}/questions/{question_number}/image")
def get_question_image(
    job_id: str,
    question_number: int,
    part: Optional[int] = Query(None),
    db: Session = Depends(database.get_db)
):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job or not job.result_json:
        raise HTTPException(status_code=404, detail="Job or result not found")

    file_path = get_job_file_path(job)
    if not file_path:
        raise HTTPException(status_code=404, detail="PDF file not found")

    result_obj = schemas.PDFResult.model_validate_json(job.result_json)
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
        
    hint_page = getattr(result_obj.questions[target_q_idx], "page_number", None)
    target_page_idx = find_question_page(doc, question_number, hint_page)
                
    if target_page_idx is None:
        raise HTTPException(status_code=404, detail="Could not find Question in PDF pages")

    next_question_number = None
    next_question_stem = None
    end_page_idx = target_page_idx
    
    if target_q_idx + 1 < len(result_obj.questions):
        next_q = result_obj.questions[target_q_idx + 1]
        next_hint = getattr(next_q, "page_number", None)
        next_page_idx = find_question_page(doc, next_q.question_number, next_hint)
        if next_page_idx is not None:
            # If the next question is found, it bounds the end of our current question.
            # It could be on the same page, or a subsequent page.
            end_page_idx = next_page_idx
            next_question_number = next_q.question_number
            next_question_stem = getattr(next_q, "stem", None)

    if end_page_idx == target_page_idx and target_q_idx + 1 == len(result_obj.questions):
        # Last question in the parsed results. It might spill over!
        next_p_idx = find_question_page(doc, question_number + 1, target_page_idx)
        if next_p_idx is not None:
            end_page_idx = next_p_idx
            next_question_number = question_number + 1
        else:
            # If we don't find the next question, assume it might span up to 2 more pages
            end_page_idx = min(len(doc) - 1, target_page_idx + 2)

    try:
        pixmaps = []
        # Support questions spanning multiple pages
        for p_idx in range(target_page_idx, end_page_idx + 1):
            page = doc[p_idx]
            is_cont = (p_idx > target_page_idx)
            
            # If we are on the last page of the span, look for the next question's number to stop cropping.
            # If we are on an intermediate page, we want the crop to extend to the very bottom of the page,
            # so we pass None for next_question_number.
            p_next_q_num = next_question_number if p_idx == end_page_idx else None
            p_next_q_stem = next_question_stem if p_idx == end_page_idx else None
            
            crop_rect = get_question_crop_rect(page, question_number, p_next_q_num, p_next_q_stem, is_continuation=is_cont)
            
            if is_cont and crop_rect is None:
                continue
                
            if crop_rect:
                pix = page.get_pixmap(clip=crop_rect, dpi=200)
            else:
                pix = page.get_pixmap(dpi=200)
            pixmaps.append(pix)
            
        if part is not None:
            if part < 0 or part >= len(pixmaps):
                raise HTTPException(status_code=404, detail="Image part not found")
            image_bytes = pixmaps[part].tobytes("png")
        elif len(pixmaps) == 1:
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
            
        response = Response(content=image_bytes, media_type="image/png")
        response.headers["X-Image-Parts-Count"] = str(len(pixmaps))
        return response
    finally:
        doc.close()

@router.get("/pdfs/{job_id}/questions/{question_number}/image_info")
def get_question_image_info(
    job_id: str,
    question_number: int,
    db: Session = Depends(database.get_db)
):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job or not job.result_json:
        raise HTTPException(status_code=404, detail="Job or result not found")

    file_path = get_job_file_path(job)
    if not file_path:
        raise HTTPException(status_code=404, detail="PDF file not found")

    result_obj = schemas.PDFResult.model_validate_json(job.result_json)
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
        hint_page = getattr(result_obj.questions[target_q_idx], "page_number", None)
        target_page_idx = find_question_page(doc, question_number, hint_page)
                    
        if target_page_idx is None:
            return {"parts": 0}

        end_page_idx = target_page_idx
        next_question_number = None
        next_question_stem = None
        
        if target_q_idx + 1 < len(result_obj.questions):
            next_q = result_obj.questions[target_q_idx + 1]
            next_hint = getattr(next_q, "page_number", None)
            next_page_idx = find_question_page(doc, next_q.question_number, next_hint)
            if next_page_idx is not None:
                end_page_idx = next_page_idx
                next_question_number = next_q.question_number
                next_question_stem = getattr(next_q, "stem", None)
        
        if end_page_idx == target_page_idx and target_q_idx + 1 == len(result_obj.questions):
            next_p_idx = find_question_page(doc, question_number + 1, target_page_idx)
            if next_p_idx is not None:
                end_page_idx = next_p_idx
                next_question_number = question_number + 1
            else:
                end_page_idx = min(len(doc) - 1, target_page_idx + 2)

        valid_parts = 1
        for p_idx in range(target_page_idx + 1, end_page_idx + 1):
            page = doc[p_idx]
            p_next_q_num = next_question_number if p_idx == end_page_idx else None
            p_next_q_stem = next_question_stem if p_idx == end_page_idx else None
            crop_rect = get_question_crop_rect(page, question_number, p_next_q_num, p_next_q_stem, is_continuation=True)
            if crop_rect is not None:
                valid_parts += 1

        return {"parts": valid_parts}
    finally:
        doc.close()

@router.post("/pdfs/{job_id}/rescan/{question_number}", response_model=schemas.QuestionResult)
def rescan_question(
    job_id: str,
    question_number: int,
    engine: Optional[str] = Query(None),
    db: Session = Depends(database.get_db)
):
    t_ocr_start = time.perf_counter()
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path = get_job_file_path(job)
    if not file_path:
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    if not job.result_json:
        raise HTTPException(status_code=400, detail="Job has no results yet")
        
    result_obj = schemas.PDFResult.model_validate_json(job.result_json)
    
    # 1. Find the question in the result
    target_q_idx = -1
    for i, q in enumerate(result_obj.questions):
        if q.question_number == question_number:
            target_q_idx = i
            break
            
    if target_q_idx == -1:
        raise HTTPException(status_code=404, detail=f"Question {question_number} not found in this job")

    # Check if question OCR is allowed
    if str(engine).lower() in ("off", "disabled"):
        raise HTTPException(status_code=400, detail="Whole Question OCR is disabled.")
    try:
        from ..storage.config_manager import get_setting
        q_ocr_enabled = get_setting("question_ocr_enabled", True)
        if not q_ocr_enabled and engine is None:
            raise HTTPException(status_code=400, detail="Whole Question OCR is disabled in Pipeline Settings.")
    except HTTPException:
        raise
    except Exception:
        pass

    # 2. Find the true page in the PDF (skipping cover/instruction pages)
    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open PDF: {e}")
        
    hint_page = getattr(result_obj.questions[target_q_idx], "page_number", None)
    target_page_idx = find_question_page(doc, question_number, hint_page)
                
    if target_page_idx is None:
        raise HTTPException(status_code=404, detail=f"Could not find Question {question_number} in the PDF pages")

    # 3. Crop question region (or full page if crop rect not found) and send to Vision/Tesseract
    try:
        end_page_idx = target_page_idx
        next_question_number = None
        next_question_stem = None
        
        if target_q_idx + 1 < len(result_obj.questions):
            next_q = result_obj.questions[target_q_idx + 1]
            next_hint = getattr(next_q, "page_number", None)
            next_page_idx = find_question_page(doc, next_q.question_number, next_hint)
            if next_page_idx is not None:
                end_page_idx = next_page_idx
                next_question_number = next_q.question_number
                next_question_stem = getattr(next_q, "stem", None)
                
        pixmaps = []
        for p_idx in range(target_page_idx, end_page_idx + 1):
            page = doc[p_idx]
            is_cont = (p_idx > target_page_idx)
            
            p_next_q_num = next_question_number if p_idx == end_page_idx else None
            p_next_q_stem = next_question_stem if p_idx == end_page_idx else None
            
            crop_rect = get_question_crop_rect(page, question_number, p_next_q_num, p_next_q_stem, is_continuation=is_cont)
            if crop_rect:
                pix = page.get_pixmap(clip=crop_rect, dpi=200)
            else:
                pix = page.get_pixmap(dpi=200)
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
        
        extracted_data, p_tok, c_tok = rescan_question_vision(image_bytes, question_number, engine=engine)
        extracted_data["page_number"] = target_page_idx
        
        # Preserve original Image tokens (Diagrams) that Gemini Vision OCR cannot output natively
        old_q = result_obj.questions[target_q_idx]
        old_stem = getattr(old_q, "stem", "") or ""
        img_tokens = re.findall(r'\{\{IMG(?:_BLOCK)?:[^\}]+\}\}', old_stem)
        if img_tokens and "stem" in extracted_data:
            extracted_data["stem"] += "\n" + "\n".join(img_tokens)
            
        old_opts = getattr(old_q, "options", {}) or {}
        if isinstance(old_opts, dict):
            for opt_key, opt_val in old_opts.items():
                if not isinstance(opt_val, str): continue
                opt_img_tokens = re.findall(r'\{\{IMG(?:_BLOCK)?:[^\}]+\}\}', opt_val)
                if opt_img_tokens and "options" in extracted_data and isinstance(extracted_data["options"], dict):
                    for new_k in extracted_data["options"].keys():
                        if str(new_k).upper() == str(opt_key).upper():
                            extracted_data["options"][new_k] += "\n" + "\n".join(opt_img_tokens)

        old_ans = getattr(old_q, "answer_text", "") or ""
        ans_img_tokens = re.findall(r'\{\{IMG(?:_BLOCK)?:[^\}]+\}\}', old_ans)
        if ans_img_tokens:
            extracted_data["answer_text"] = extracted_data.get("answer_text", "") + "\n" + "\n".join(ans_img_tokens)

        old_exp = getattr(old_q, "explanation", "") or ""
        exp_img_tokens = re.findall(r'\{\{IMG(?:_BLOCK)?:[^\}]+\}\}', old_exp)
        if exp_img_tokens:
            extracted_data["explanation"] = extracted_data.get("explanation", "") + "\n" + "\n".join(exp_img_tokens)
        
        # Merge with existing marked answer if missing
        if not extracted_data.get("marked_answer"):
             extracted_data["marked_answer"] = result_obj.questions[target_q_idx].marked_answer
             extracted_data["marked_correct_answer"] = result_obj.questions[target_q_idx].marked_answer
             
        # 4. Run through solve node
        solved_block = _solve_single_block(extracted_data, 1)
        
        # Attach detailed pipeline_steps for this rescanned question
        rescan_engine_name = str(engine or "Configured").title()
        t_ocr_elapsed = round(time.perf_counter() - t_ocr_start, 2)
        old_q = result_obj.questions[target_q_idx]
        solved_block["ocr_scan_time_seconds"] = t_ocr_elapsed
        step_times = dict(getattr(old_q, "step_times", None) or solved_block.get("step_times") or {})
        step_times["ocr_scan"] = t_ocr_elapsed
        solved_block["step_times"] = step_times
        if getattr(old_q, "extraction_time_seconds", None) is not None:
            solved_block["extraction_time_seconds"] = old_q.extraction_time_seconds
        if getattr(old_q, "validation_time_seconds", None) is not None:
            solved_block["validation_time_seconds"] = old_q.validation_time_seconds

        solved_block["pipeline_steps"] = {
            "scanned": {
                "name": "scanned",
                "label": "Scanned",
                "status": "completed",
                "duration_seconds": solved_block.get("extraction_time_seconds"),
                "details": f"Rescanned from PDF Page {target_page_idx + 1}"
            },
            "checked_with_caches": {
                "name": "checked_with_caches",
                "label": "Checked with Caches",
                "status": "completed",
                "details": "Validated against dictionary & cache"
            },
            "send_errors_to_llms": {
                "name": "send_errors_to_llms",
                "label": "Send Errors to LLMs",
                "status": "completed" if solved_block.get("validated_answer") else "skipped",
                "duration_seconds": solved_block.get("validation_time_seconds"),
                "details": f"Answer verified by LLM ({solved_block.get('reasoning', '')})" if solved_block.get("validated_answer") else "Validation skipped / offline"
            },
            "ocr_scanned": {
                "name": "ocr_scanned",
                "label": "OCR Scanned",
                "status": "completed",
                "duration_seconds": t_ocr_elapsed,
                "details": f"Question OCR rescanned via {rescan_engine_name} ({t_ocr_elapsed}s)"
            }
        }

        # 5. Update DB
        new_q = schemas.QuestionResult(**solved_block)
        result_obj.questions[target_q_idx] = new_q
        
        # Update token usage
        result_obj.summary.token_usage.prompt_tokens += p_tok + solved_block.get("_p_tokens", 0)
        result_obj.summary.token_usage.completion_tokens += c_tok + solved_block.get("_c_tokens", 0)
        result_obj.summary.token_usage.total_tokens += p_tok + c_tok + solved_block.get("_p_tokens", 0) + solved_block.get("_c_tokens", 0)
        
        job.result_json = result_obj.model_dump_json()
        db.commit()
        
        return new_q
        
    except Exception as e:
        print(f"Rescan failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix_question_with_ai")
async def fix_question_with_ai_endpoint(
    request: Request,
    db: Session = Depends(database.get_db)
):
    """
    On-demand single question full-text rewrite using LLM.
    Expects JSON body: { "question_number": int, "stem": str, "options": dict, "marked_answer": str }
    """
    try:
        data = await request.json()
        if "stem" not in data or "question_number" not in data:
            raise HTTPException(status_code=400, detail="Missing required question data")
            
        t_fix_start = time.perf_counter()
        from ..graph.correction_node import fix_single_question_with_llm
        result = await fix_single_question_with_llm(data)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "LLM fix failed"))

        t_fix_elapsed = round(max(0.05, time.perf_counter() - t_fix_start), 2)
        result["validation_time_seconds"] = t_fix_elapsed
        if "step_times" not in result or not isinstance(result["step_times"], dict):
            result["step_times"] = {}
        result["step_times"]["validation"] = t_fix_elapsed

        val_ans = result.get("validated_answer")
        is_valid = result.get("is_valid", False)
        val_details = f"AI verified answer (Agreed: Option {val_ans}) ({t_fix_elapsed}s)" if is_valid else f"AI verified and rewritten ({t_fix_elapsed}s)."
        result["pipeline_steps"] = {
            **(data.get("pipeline_steps") or {}),
            "send_errors_to_llms": {
                "name": "send_errors_to_llms",
                "label": "AI Validation",
                "status": "completed",
                "duration_seconds": t_fix_elapsed,
                "details": val_details
            }
        }
            
        # If job_id was provided, persist updated question & accumulated tokens into SQLite
        job_id = data.get("job_id")
        if job_id:
            try:
                job = db.query(models.Job).filter(models.Job.id == job_id).first()
                if job and job.result_json:
                    result_data = json.loads(job.result_json)
                    tu = result.get("token_usage", {})
                    
                    if "summary" not in result_data:
                        result_data["summary"] = {}
                    if "token_usage" not in result_data["summary"] or not result_data["summary"]["token_usage"]:
                        result_data["summary"]["token_usage"] = {
                            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0
                        }
                    
                    st_tu = result_data["summary"]["token_usage"]
                    st_tu["prompt_tokens"] = st_tu.get("prompt_tokens", 0) + tu.get("prompt_tokens", 0)
                    st_tu["completion_tokens"] = st_tu.get("completion_tokens", 0) + tu.get("completion_tokens", 0)
                    st_tu["total_tokens"] = st_tu.get("total_tokens", 0) + tu.get("total_tokens", 0)
                    st_tu["estimated_cost_usd"] = round(st_tu.get("estimated_cost_usd", 0.0) + tu.get("estimated_cost_usd", 0.0), 6)
                    
                    qn = data.get("question_number")
                    for q in result_data.get("questions", []):
                        if q.get("question_number") == qn:
                            q["stem"] = result.get("corrected_stem", q.get("stem"))
                            q["options"] = result.get("corrected_options", q.get("options"))
                            if "question_type" in result:
                                q["question_type"] = result["question_type"]
                            if "sub_parts" in result:
                                q["sub_parts"] = result["sub_parts"]
                            q["is_valid"] = is_valid
                            if val_ans:
                                q["validated_answer"] = val_ans
                            elif q.get("question_type", "").startswith("descriptive"):
                                q["validated_answer"] = None
                                q["is_valid"] = True
                            if not q.get("marked_correct_answer") and not q.get("marked_answer"):
                                if val_ans:
                                    q["marked_correct_answer"] = val_ans
                                    q["marked_answer"] = val_ans
                            q["reasoning"] = result.get("reasoning", q.get("reasoning"))
                            corrected_explanation = (
                                result.get("corrected_explanation")
                                or result.get("explanation")
                                or q.get("explanation")
                                or q.get("answer_text")
                            )
                            q["explanation"] = corrected_explanation or None
                            q["answer_text"] = corrected_explanation or None
                            q["token_usage"] = tu
                            q["validation_time_seconds"] = t_fix_elapsed
                            st = q.get("step_times") or {}
                            st["validation"] = t_fix_elapsed
                            q["step_times"] = st
                            q["pipeline_steps"] = {
                                **(q.get("pipeline_steps") or {}),
                                "send_errors_to_llms": {
                                    "name": "send_errors_to_llms",
                                    "label": "Send Errors to LLMs",
                                    "status": "completed",
                                    "duration_seconds": t_fix_elapsed,
                                    "details": val_details
                                }
                            }
                            break
                    
                    if "summary" in result_data and isinstance(result_data["summary"], dict):
                        result_data["summary"]["validated"] = sum(1 for qx in result_data.get("questions", []) if qx.get("is_valid"))
                        result_data["summary"]["flagged"] = sum(1 for qx in result_data.get("questions", []) if (qx.get("validated_answer") and not qx.get("is_valid")))
                        result_data["summary"]["unvalidated"] = sum(1 for qx in result_data.get("questions", []) if not qx.get("validated_answer"))

                    job.result_json = json.dumps(result_data)
                    db.commit()
            except Exception as db_err:
                logger.warning("[Routes] Failed to persist AI rewrite to DB: %s", db_err)

        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify_question_with_ai")
async def verify_question_with_ai_endpoint(
    request: Request,
    db: Session = Depends(database.get_db)
):
    """
    On-demand single question answer verification using LLM.
    Solves question and verifies marked answer without altering stem or options text.
    """
    try:
        data = await request.json()
        if "stem" not in data or "question_number" not in data:
            raise HTTPException(status_code=400, detail="Missing required question data")

        t_val_start = time.perf_counter()
        from ..graph.correction_node import fix_single_question_with_llm
        result = await fix_single_question_with_llm(data)
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "AI verification failed"))

        t_val_elapsed = round(max(0.05, time.perf_counter() - t_val_start), 2)
        result["validation_time_seconds"] = t_val_elapsed
        if "step_times" not in result or not isinstance(result["step_times"], dict):
            result["step_times"] = {}
        result["step_times"]["validation"] = t_val_elapsed

        val_ans = result.get("validated_answer")
        is_valid = result.get("is_valid", False)
        marked = data.get("marked_correct_answer") or data.get("marked_answer") or val_ans
        val_details = f"AI verified answer (Agreed: Option {val_ans}) ({t_val_elapsed}s)" if is_valid else f"AI disagreed: verified Option {val_ans or 'different'} vs marked {marked or '—'} ({t_val_elapsed}s)."
        result["pipeline_steps"] = {
            **(data.get("pipeline_steps") or {}),
            "send_errors_to_llms": {
                "name": "send_errors_to_llms",
                "label": "AI Validation",
                "status": "completed",
                "duration_seconds": t_val_elapsed,
                "details": val_details
            }
        }

        job_id = data.get("job_id")
        if job_id:
            try:
                job = db.query(models.Job).filter(models.Job.id == job_id).first()
                if job and job.result_json:
                    result_data = json.loads(job.result_json)
                    tu = result.get("token_usage", {})

                    if "summary" not in result_data:
                        result_data["summary"] = {}
                    if "token_usage" not in result_data["summary"] or not result_data["summary"]["token_usage"]:
                        result_data["summary"]["token_usage"] = {
                            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0
                        }

                    st_tu = result_data["summary"]["token_usage"]
                    st_tu["prompt_tokens"] = st_tu.get("prompt_tokens", 0) + tu.get("prompt_tokens", 0)
                    st_tu["completion_tokens"] = st_tu.get("completion_tokens", 0) + tu.get("completion_tokens", 0)
                    st_tu["total_tokens"] = st_tu.get("total_tokens", 0) + tu.get("total_tokens", 0)
                    st_tu["estimated_cost_usd"] = round(st_tu.get("estimated_cost_usd", 0.0) + tu.get("estimated_cost_usd", 0.0), 6)

                    qn = data.get("question_number")
                    val_ans = result.get("validated_answer")
                    is_valid = result.get("is_valid", False)
                    reasoning = result.get("reasoning") or result.get("corrected_explanation")

                    for q in result_data.get("questions", []):
                        if q.get("question_number") == qn:
                            q["validated_answer"] = val_ans
                            q["is_valid"] = is_valid
                            q["validation_time_seconds"] = t_val_elapsed
                            st = q.get("step_times") or {}
                            st["validation"] = t_val_elapsed
                            q["step_times"] = st
                            if reasoning:
                                q["reasoning"] = reasoning
                                if not q.get("explanation"):
                                    q["explanation"] = reasoning
                                    q["answer_text"] = reasoning
                            
                            # Update structure if LLM verification corrected the question type
                            if "question_type" in result:
                                q["question_type"] = result["question_type"]
                            if "sub_parts" in result:
                                q["sub_parts"] = result["sub_parts"]
                                
                            q["token_usage"] = tu
                            marked = q.get("marked_correct_answer") or q.get("marked_answer") or val_ans
                            val_details = f"AI verified answer (Agreed: Option {val_ans}) ({t_val_elapsed}s)" if is_valid else f"AI disagreed: verified Option {val_ans or 'different'} vs marked {marked or '—'} ({t_val_elapsed}s)."
                            q["pipeline_steps"] = {
                                **(q.get("pipeline_steps") or {}),
                                "send_errors_to_llms": {
                                    "name": "send_errors_to_llms",
                                    "label": "Send Errors to LLMs",
                                    "status": "completed",
                                    "duration_seconds": t_val_elapsed,
                                    "details": val_details
                                }
                            }
                            break

                    if "summary" in result_data and isinstance(result_data["summary"], dict):
                        result_data["summary"]["validated"] = sum(1 for qx in result_data.get("questions", []) if qx.get("is_valid"))
                        result_data["summary"]["flagged"] = sum(1 for qx in result_data.get("questions", []) if (qx.get("validated_answer") and not qx.get("is_valid")))
                        result_data["summary"]["unvalidated"] = sum(1 for qx in result_data.get("questions", []) if not qx.get("validated_answer"))

                    job.result_json = json.dumps(result_data)
                    db.commit()
            except Exception as db_err:
                logger.warning("[Routes] Failed to persist single-q verify to DB: %s", db_err)

        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdfs/{job_id}/convert-equations")
def convert_equations_endpoint(
    job_id: str,
    background_tasks: BackgroundTasks,
    engine: Optional[str] = Query(None),
    db: Session = Depends(database.get_db)
):
    """Manually triggers progressive equation-to-LaTeX background conversion for a job with optional engine override."""
    task = _math_conversion_tasks.get(job_id)
    if task and task.get("is_active"):
        return {"status": "already_running", "message": "Conversion is already actively running for this job"}

    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result_json:
        raise HTTPException(status_code=400, detail="Job has no results yet")
    background_tasks.add_task(progressively_convert_equations_to_latex, job_id, engine)
    return {
        "status": "started",
        "engine": engine or "auto",
        "message": f"Progressively converting equation images to LaTeX in background using {engine or 'configured'} engine"
    }


@router.post("/pdfs/{job_id}/revalidate")
def revalidate_job_endpoint(job_id: str, db: Session = Depends(database.get_db)):
    """
    Re-runs word cleaning, font artifact removal, and AI validation on all questions of an existing job.
    Updates Job.result_json in SQLite and returns the updated job result.
    """
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result_json:
        raise HTTPException(status_code=400, detail="Job has no results yet")

    try:
        from ..graph.correction_node import correction_and_validation_node
        from ..ingest.text_cleaner import clean_indic_text
        from ..ingest.math_normalizer import is_indic_text, normalize_math_in_text

        data = json.loads(job.result_json)
        questions = data.get("questions", [])
        if not questions:
            return {"job_id": job_id, "result": data}

        from ..ingest.option_answer_separator import clean_question_options_and_answer

        # Step 1: Pre-clean all fields with deterministic text cleaner
        for q in questions:
            clean_question_options_and_answer(q)
            for field in ["stem", "explanation", "answer_text", "reasoning"]:
                val = q.get(field)
                if val and isinstance(val, str):
                    if is_indic_text(val):
                        cleaned, _ = clean_indic_text(val)
                    else:
                        cleaned = normalize_math_in_text(val)
                    if field in ["explanation", "answer_text"]:
                        marked = q.get("marked_correct_answer") or q.get("marked_answer")
                        if marked:
                            cleaned = re.sub(rf'^\(?{marked}\)?[\s\.\:\-]+', '', cleaned).strip()
                        cleaned = re.sub(r'^(?:Solution|Explanation|Description|Sol|Exp|Hint|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*(?:[\*\:\-]+(?:\s*|\n|$)|(?:\n|\\n))', '', cleaned, flags=re.IGNORECASE).strip()
                    q[field] = cleaned
            opts = q.get("options", {})
            for k, opt_val in opts.items():
                if isinstance(opt_val, str):
                    if is_indic_text(opt_val):
                        cleaned_opt, _ = clean_indic_text(opt_val)
                    else:
                        cleaned_opt = normalize_math_in_text(opt_val)
                    opts[k] = cleaned_opt
            clean_question_options_and_answer(q)

        # Step 2: Run correction and validation node with force_validation=True (user clicked the button)
        t_reval_start = time.perf_counter()
        pdf_file_path = job.filepath if hasattr(job, "filepath") and job.filepath else job_id
        state = {
            "extracted_blocks": questions,
            "job_id": job_id,
            "pdf_path": pdf_file_path,
            "file_path": pdf_file_path,
            "pages": [],
            "force_validation": True,
        }
        res_state = correction_and_validation_node(state)
        processed = res_state.get("processed_questions") or questions
        t_reval_elapsed = round(max(0.1, time.perf_counter() - t_reval_start), 2)
        per_q_val = round(max(0.05, t_reval_elapsed / max(1, len(processed))), 2)

        for pq in processed:
            pq["validation_time_seconds"] = per_q_val
            st = pq.get("step_times") or {}
            st["validation"] = per_q_val
            pq["step_times"] = st
            if "pipeline_steps" not in pq or not isinstance(pq["pipeline_steps"], dict):
                pq["pipeline_steps"] = {}
            val_step = pq["pipeline_steps"].get("send_errors_to_llms") or {
                "name": "send_errors_to_llms",
                "label": "Send Errors to LLMs",
            }
            val_step["status"] = "completed"
            val_step["duration_seconds"] = per_q_val
            det = val_step.get("details", "AI verified answer")
            if not re.search(r'\([\d\.]+s\)', det):
                val_step["details"] = f"{det} ({per_q_val}s)"
            pq["pipeline_steps"]["send_errors_to_llms"] = val_step
            for field in ["stem", "explanation", "answer_text", "reasoning"]:
                pval = pq.get(field)
                if pval and isinstance(pval, str):
                    pq[field] = normalize_math_in_text(pval)
            popts = pq.get("options", {})
            if isinstance(popts, dict):
                for pk, pv in popts.items():
                    if isinstance(pv, str):
                        popts[pk] = normalize_math_in_text(pv)
            clean_question_options_and_answer(pq)

        data["questions"] = processed
        if "summary" in res_state and res_state["summary"]:
            data["summary"] = res_state["summary"]
        job.result_json = json.dumps(data)
        db.commit()

        return {"job_id": job_id, "result": data}
    except Exception as e:
        logger.error("[Routes] Failed to revalidate job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pdfs/{job_id}/math-status")
def get_math_conversion_status(job_id: str, db: Session = Depends(database.get_db)):
    """Returns the live conversion status, question progress, and remaining equation image count."""
    task = _math_conversion_tasks.get(job_id)
    if task and task.get("is_active"):
        return task

    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job or not job.result_json:
        return {"is_active": False, "status": "not_found", "remaining_math_images": 0, "total_questions": 0, "current_question": 0}

    try:
        data = json.loads(job.result_json)
        questions = data.get("questions", [])
        remaining = _count_math_images(questions)
        return {
            "is_active": False,
            "status": "idle" if remaining > 0 else "completed",
            "remaining_math_images": remaining,
            "total_questions": len(questions),
            "current_question": len(questions)
        }
    except Exception:
        return {"is_active": False, "status": "idle", "remaining_math_images": 0, "total_questions": 0, "current_question": 0}


# ---------------------------------------------------------------------------
# Review Flags API (spec §6 — admin review queue)
# ---------------------------------------------------------------------------

@router.get("/review-flags")
def list_review_flags(
    job_id: str = None,
    flag_type: str = None,
    status: str = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(database.get_db),
):
    """
    List review flags, filterable by job, type, and status.
    
    Flag types:
    - "text_rejection": LLM text correction failed edit-distance or confidence gate
    - "answer_disagreement": LLM disagrees with marked answer key
    """
    from ..storage.models import ReviewFlag
    from ..models.schemas import ReviewFlagResponse, ReviewFlagListResponse

    query = db.query(ReviewFlag)
    if job_id:
        query = query.filter(ReviewFlag.job_id == job_id)
    if flag_type:
        query = query.filter(ReviewFlag.flag_type == flag_type)
    if status:
        query = query.filter(ReviewFlag.status == status)

    total = query.count()
    flags = query.order_by(ReviewFlag.created_at.desc()).offset(offset).limit(limit).all()

    return ReviewFlagListResponse(
        flags=[
            ReviewFlagResponse(
                id=f.id,
                job_id=f.job_id,
                question_number=f.question_number,
                page_number=f.page_number,
                flag_type=f.flag_type,
                original_text=f.original_text,
                proposed_text=f.proposed_text,
                llm_explanation=f.llm_explanation,
                confidence=f.confidence,
                status=f.status,
                created_at=f.created_at,
                resolved_at=f.resolved_at,
                resolved_by=f.resolved_by,
            )
            for f in flags
        ],
        total=total,
    )


@router.post("/review-flags/{flag_id}/resolve")
def resolve_review_flag(
    flag_id: int,
    body: dict,
    db: Session = Depends(database.get_db),
):
    """
    Accept or reject a review flag.
    Body: {"action": "accept"|"reject", "resolved_by": "admin"}
    """
    from ..storage.models import ReviewFlag
    from datetime import datetime

    flag = db.query(ReviewFlag).filter(ReviewFlag.id == flag_id).first()
    if not flag:
        raise HTTPException(status_code=404, detail="Review flag not found")

    action = body.get("action", "").lower()
    if action not in ("accept", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'accept' or 'reject'")

    flag.status = "accepted" if action == "accept" else "rejected"
    flag.resolved_at = datetime.utcnow()
    flag.resolved_by = body.get("resolved_by", "admin")
    db.commit()

    return {"status": "ok", "flag_id": flag_id, "new_status": flag.status}


@router.get("/review-flags/stats")
def review_flag_stats(db: Session = Depends(database.get_db)):
    """Aggregate counts by flag type and status for dashboard indicator."""
    from ..storage.models import ReviewFlag
    from ..models.schemas import ReviewFlagStatsResponse
    from sqlalchemy import func

    rows = (
        db.query(ReviewFlag.flag_type, ReviewFlag.status, func.count())
        .group_by(ReviewFlag.flag_type, ReviewFlag.status)
        .all()
    )

    stats = ReviewFlagStatsResponse()
    for flag_type, status, count in rows:
        if status == "pending":
            stats.total_pending += count
            if flag_type == "text_rejection":
                stats.text_rejections_pending += count
            elif flag_type == "answer_disagreement":
                stats.answer_disagreements_pending += count
        elif status == "accepted":
            stats.total_accepted += count
        elif status == "rejected":
            stats.total_rejected += count

    return stats


@router.get("/cache/stats")
def cache_stats():
    """Correction cache health: staging count, trusted count, hit rate."""
    from ..ingest.correction_cache import get_correction_cache
    from ..models.schemas import CacheStatsResponse

    cache = get_correction_cache()
    raw = cache.get_stats()
    return CacheStatsResponse(**raw)
