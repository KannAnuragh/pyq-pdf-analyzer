from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException
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
from ..graph.nodes import get_gemini_status, is_gemini_enabled, get_groq_status, is_groq_enabled, get_llm_status, is_llm_enabled, is_answer_validation_enabled, rescan_question_vision, _solve_single_block, separate_stem_and_options, parse_trailing_meta

_math_conversion_tasks: dict[str, dict] = {}

def _count_math_images(questions: list) -> int:
    """Counts remaining equation images in questions (excluding genuine geometry diagrams)."""
    cnt = 0
    for q in questions:
        for text in [q.get("stem", "")] + list(q.get("options", {}).values()):
            if '<img' in text:
                for m in re.finditer(r'<img[^>]*>', text):
                    tag = m.group(0)
                    if 'alt="diagram"' not in tag:
                        cnt += 1
    return cnt

def _replace_math_images_in_text(text: str) -> tuple[str, bool]:
    """Replaces equation and diagram image tags with LaTeX by running targeted OCR on each clip."""
    if not text or '<img' not in text:
        return text, False

    # Match both standalone <img ... /> and wrapper <div ...><img ... /></div>
    pattern = r'(?:<div[^>]*>\s*)?<img[^>]*src="data:image/png;base64,([^"]+)"[^>]*\/?>\s*(?:<\/div>)?'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return text, False

    modified = False
    new_text = text
    for m in reversed(matches):
        tag_str = m.group(0)
        # Skip genuine geometry/drawing diagrams (e.g. alt="diagram" with large size)
        if 'alt="diagram"' in tag_str and 'inline-math-img' not in tag_str:
            try:
                b64_chk = m.group(1).strip()
                pad = len(b64_chk) % 4
                if pad:
                    b64_chk += '=' * (4 - pad)
                img_bytes_chk = base64.b64decode(b64_chk)
                from PIL import Image
                chk_img = Image.open(io.BytesIO(img_bytes_chk))
                cw, ch = chk_img.size
                if cw > 180 and ch > 120:
                    continue
            except Exception:
                pass

        b64_str = m.group(1).strip()
        try:
            pad = len(b64_str) % 4
            if pad:
                b64_str += '=' * (4 - pad)
            img_bytes = base64.b64decode(b64_str)
            latex, _, _ = ocr_equation_clip(img_bytes)
            clean_math = clean_latex_equation(latex)
            if clean_math and clean_math != "$$":
                new_text = new_text[:m.start()] + f" {clean_math} " + new_text[m.end():]
                modified = True
        except Exception as e:
            print(f"[Progressive Math OCR] Tag parse failed: {e}", flush=True)

    if modified:
        new_text = re.sub(r' +', ' ', new_text).strip()

    return new_text, modified


def progressively_convert_equations_to_latex(job_id: str):
    """
    Progressively scans equation images in background and converts them to LaTeX.
    Runs persistently across passes until every single equation is converted.
    Updates the database per-question so the frontend live-updates in real-time.
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

        print(f"[Progressive Math OCR] Starting background LaTeX conversion for Job {job_id} ({initial_remaining} equation clips)...", flush=True)

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
                new_stem, mod_stem = _replace_math_images_in_text(q.get("stem", ""))
                if mod_stem:
                    q["stem"] = new_stem
                    updated_q = True

                # Options
                opts = q.get("options", {})
                for k, v in list(opts.items()):
                    new_v, mod_v = _replace_math_images_in_text(v)
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

def process_pdf_background(job_id: str, file_path: str, db: Session):
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
        ingest_p_tokens = ingest_result["prompt_tokens"]
        ingest_c_tokens = ingest_result["completion_tokens"]
        
        # 2. LangGraph Workflow Streaming
        initial_state = GraphState(
            file_path=file_path,
            raw_text=raw_text,
            pages=pages_metadata,
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
            
            # 3. Intermediate DB Update (when extraction is done, but before slow validation)
            if node_name == "rule_check":
                job = db.query(models.Job).filter(models.Job.id == job_id).first()
                if job:
                    questions = []
                    for b in current_full_state.get("extracted_blocks", []):
                        # Show raw blocks immediately
                        marked = b.get("marked_correct_answer") or b.get("marked_answer")
                        questions.append(schemas.QuestionResult(
                            question_number=b.get("question_number", 0),
                            stem=b.get("stem", ""),
                            options=b.get("options", {}),
                            marked_answer=marked,
                            marked_correct_answer=marked
                        ))
                        
                    summary_data = current_full_state.get("summary", {"total": len(questions), "validated": 0, "flagged": 0})
                    token_usage_data = current_full_state.get("token_usage", {})
                    summary_data["token_usage"] = schemas.TokenUsage(
                        prompt_tokens=token_usage_data.get("prompt_tokens", 0),
                        completion_tokens=token_usage_data.get("completion_tokens", 0),
                        total_tokens=token_usage_data.get("total_tokens", 0)
                    )
                    
                    result = schemas.PDFResult(
                        source_file=job.filename,
                        processed_at=job.created_at,
                        questions=questions,
                        summary=schemas.Summary(**summary_data)
                    )
                    
                    job.status = "validating"
                    job.result_json = result.model_dump_json()
                    db.commit()

        # 4. Final DB Update
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if job:
            questions = []
            for pq in current_full_state.get("processed_questions", []):
                marked = pq.get("marked_correct_answer") or pq.get("marked_answer")
                pq["marked_correct_answer"] = marked
                pq["marked_answer"] = marked
                questions.append(schemas.QuestionResult(**pq))
                
            summary_data = current_full_state.get("summary", {"total": 0, "validated": 0, "flagged": 0})
            token_usage_data = current_full_state.get("token_usage", {})
            summary_data["token_usage"] = schemas.TokenUsage(
                prompt_tokens=token_usage_data.get("prompt_tokens", 0),
                completion_tokens=token_usage_data.get("completion_tokens", 0),
                total_tokens=token_usage_data.get("total_tokens", 0)
            )
            
            result = schemas.PDFResult(
                source_file=job.filename,
                processed_at=job.created_at,
                questions=questions,
                summary=schemas.Summary(**summary_data)
            )
            
            job.status = "done"
            job.result_json = result.model_dump_json()
            db.commit()

            # Slowly convert equation images to LaTeX in the background
            threading.Thread(
                target=progressively_convert_equations_to_latex,
                args=(job_id,),
                daemon=True
            ).start()
            
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if job:
            job.status = "failed"
            db.commit()

@router.post("/pdfs/upload", response_model=schemas.JobStatusResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Create job entry
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

    # Dispatch background task
    # Note: we need to pass a new db session to the background task usually, or handle it carefully
    background_tasks.add_task(process_pdf_background, job.id, file_path, database.SessionLocal())

    return schemas.JobStatusResponse(
        job_id=job.id,
        status=job.status,
        filename=job.filename,
        created_at=job.created_at,
        has_file=True
    )

@router.get("/pdfs/{job_id}/status", response_model=schemas.JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(database.get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path = get_job_file_path(job)

    return schemas.JobStatusResponse(
        job_id=job.id,
        status=job.status,
        filename=job.filename,
        created_at=job.created_at,
        has_file=file_path is not None
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
        # Self-healing: if any question has empty options or trapped answer/explanation
        changed = False
        for q in result_obj.questions:
            # 1. Trapped in stem when options empty or incomplete (< 4 options while trapped markers exist)
            re_separate_needed = False
            if not q.options or len(q.options) < 4:
                # Check if options or stem contain unseparated option markers like B), C), D)
                combined_probe = (q.stem or "") + " " + " ".join(q.options.values() if q.options else [])
                if any(m in combined_probe for m in ["B)", "C)", "D)", "B:-", "C:-", "D:-", "B.", "C.", "D."]):
                    re_separate_needed = True

            if re_separate_needed:
                # Reconstruct full block to reliably extract all options A, B, C, D
                reconstruct_text = (q.stem or "").strip()
                if q.options:
                    for k in sorted(q.options.keys()):
                        reconstruct_text += f"\n{k}) {q.options[k]}"
                if q.marked_answer and "answer" not in reconstruct_text.lower() and "ഉത്തരം" not in reconstruct_text:
                    reconstruct_text += f"\nAnswer: {q.marked_answer}"
                if q.explanation and "solution" not in reconstruct_text.lower() and "വിശദീകരണം" not in reconstruct_text:
                    reconstruct_text += f"\nSolution: {q.explanation}"

                healed_stem, healed_options, healed_ans, healed_exp = separate_stem_and_options(reconstruct_text)
                if len(healed_options) > len(q.options or {}):
                    q.stem = healed_stem
                    q.options = healed_options
                    if not q.marked_answer and healed_ans:
                        q.marked_answer = healed_ans
                        q.marked_correct_answer = healed_ans
                    if not q.explanation and healed_exp:
                        q.explanation = healed_exp
                    changed = True

            # 2. Trapped answer or explanation in the last option (e.g. Option D has Answer / Solution / ഉത്തരം / വിശദീകരണം)
            if q.options:
                last_key = sorted(q.options.keys())[-1]
                last_val = q.options[last_key]
                clean_opt, ans, exp = parse_trailing_meta(last_val)
                if clean_opt != last_val:
                    q.options[last_key] = clean_opt
                    changed = True
                if ans and not q.marked_answer:
                    q.marked_answer = ans
                    q.marked_correct_answer = ans
                    changed = True
                if exp and not q.explanation:
                    q.explanation = exp
                    changed = True

        if changed:
            try:
                job.result_json = result_obj.model_dump_json()
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"Warning: Failed to persist self-healed options/explanation for job {job_id}: {e}")

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
        results.append(
            schemas.JobStatusResponse(
                job_id=j.id,
                status=j.status,
                filename=j.filename,
                created_at=j.created_at,
                has_file=has_file
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
    gemini_st = get_gemini_status()
    groq_st = get_groq_status()
    llm_st = get_llm_status()
    return {
        "gemini_enabled": gemini_st["enabled"],
        "gemini_status": gemini_st,
        "groq_enabled": groq_st["enabled"],
        "groq_status": groq_st,
        "llm_enabled": llm_st["enabled"],
        "active_provider": llm_st.get("provider", "none"),
        "answer_validation_enabled": is_answer_validation_enabled()
    }

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


def get_question_crop_rect(page, question_number: int):
    """Calculates tight bounding box Rect for the specified question on page."""
    blocks = page.get_text("blocks")
    start_y = None
    end_y = None
    
    q_start_pat = re.compile(fr'(?:^|\n)\s*{question_number}\s*[\.\):-]\s+')
    next_q_pat = re.compile(fr'(?:^|\n)\s*{question_number + 1}\s*[\.\):-]\s+')
    
    for b in blocks:
        txt = b[4]
        if start_y is None and q_start_pat.search(txt):
            start_y = max(0, b[1] - 12)
        elif start_y is not None and next_q_pat.search(txt):
            end_y = b[1] - 5
            break
            
    if start_y is not None:
        if end_y is None:
            end_y = max(start_y + 100, page.rect.height - 30)
        return pymupdf.Rect(0, start_y, page.rect.width, end_y)
    return None


@router.post("/pdfs/{job_id}/rescan/{question_number}", response_model=schemas.QuestionResult)
def rescan_question(job_id: str, question_number: int, db: Session = Depends(database.get_db)):
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

    # 2. Find the true page in the PDF (skipping cover/instruction pages)
    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open PDF: {e}")
        
    hint_page = getattr(result_obj.questions[target_q_idx], "page_number", None)
    target_page_idx = find_question_page(doc, question_number, hint_page)
                
    if target_page_idx is None:
        raise HTTPException(status_code=404, detail=f"Could not find Question {question_number} in the PDF pages")

    # 3. Crop question region (or full page if crop rect not found) and send to Vision
    try:
        page = doc[target_page_idx]
        crop_rect = get_question_crop_rect(page, question_number)
        if crop_rect:
            pix = page.get_pixmap(clip=crop_rect, dpi=200)
        else:
            pix = page.get_pixmap(dpi=200)
        image_bytes = pix.tobytes("png")
        
        extracted_data, p_tok, c_tok = rescan_question_vision(image_bytes, question_number)
        extracted_data["page_number"] = target_page_idx
        
        # Merge with existing marked answer if missing
        if not extracted_data.get("marked_answer"):
             extracted_data["marked_answer"] = result_obj.questions[target_q_idx].marked_answer
             extracted_data["marked_correct_answer"] = result_obj.questions[target_q_idx].marked_answer
             
        # 4. Run through solve node
        solved_block = _solve_single_block(extracted_data, 1)
        
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


@router.post("/pdfs/{job_id}/convert-equations")
def convert_equations_endpoint(job_id: str, background_tasks: BackgroundTasks, db: Session = Depends(database.get_db)):
    """Manually triggers progressive equation-to-LaTeX background conversion for a job."""
    task = _math_conversion_tasks.get(job_id)
    if task and task.get("is_active"):
        return {"status": "already_running", "message": "Conversion is already actively running for this job"}

    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result_json:
        raise HTTPException(status_code=400, detail="Job has no results yet")
    background_tasks.add_task(progressively_convert_equations_to_latex, job_id)
    return {"status": "started", "message": "Progressively converting equation images to LaTeX in background"}


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


