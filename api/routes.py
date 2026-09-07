from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import json
import os
import urllib.parse

import pymupdf
import re

from ..storage import database, models
from ..models import schemas

router = APIRouter()

from ..ingest.router import route_and_extract
from ..graph.workflow import app_graph
from ..graph.state import GraphState
from ..graph.nodes import get_gemini_status, is_gemini_enabled, rescan_question_vision, _solve_single_block

def process_pdf_background(job_id: str, file_path: str, db: Session):
    try:
        # Mark as processing
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if job:
            job.status = "processing"
            db.commit()

        # 1. Ingestion
        raw_text, ingest_p_tokens, ingest_c_tokens = route_and_extract(file_path)
        
        # 2. LangGraph Workflow Streaming
        initial_state = GraphState(
            file_path=file_path,
            raw_text=raw_text,
            extracted_blocks=[],
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
            
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        job = db.query(models.Job).filter(models.Job.id == job_id).first()
        if job:
            job.status = "failed"
            db.commit()

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
    status = get_gemini_status()
    return {
        "gemini_enabled": status["enabled"],
        "gemini_status": status
    }

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

    # 2. Find the page in the PDF
    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open PDF: {e}")
        
    q_pattern_explicit = re.compile(fr'(?i)(?:^|\n)\s*(?:Question\s*|Q\.?\s*){question_number}\s*(?::\-?|[\.\):-])\s*')
    q_pattern_fallback = re.compile(fr'(?:^|\n)\s*{question_number}\s*(?:[\.\)]|:-)\s+')
    
    target_page_idx = None
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text()
        if q_pattern_explicit.search(text):
            target_page_idx = page_idx
            break
            
    if target_page_idx is None:
        for page_idx in range(len(doc)):
            text = doc[page_idx].get_text()
            if q_pattern_fallback.search(text):
                target_page_idx = page_idx
                break
                
    if target_page_idx is None:
        raise HTTPException(status_code=404, detail="Could not find this question in the PDF pages")

    # 3. Render page and send to Gemini Vision
    try:
        page = doc[target_page_idx]
        pix = page.get_pixmap(dpi=200)
        image_bytes = pix.tobytes("png")
        
        extracted_data, p_tok, c_tok = rescan_question_vision(image_bytes, question_number)
        
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

