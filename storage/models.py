from sqlalchemy import Column, String, DateTime, Text, Integer, Float, ForeignKey, UniqueConstraint, Boolean
from .database import Base
from datetime import datetime
import uuid

class Client(Base):
    __tablename__ = "clients"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    daily_upload_limit = Column(Integer, nullable=True, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False, index=True)
    key_hash = Column(String, nullable=False, unique=True, index=True)
    prefix = Column(String, nullable=False)
    name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String, nullable=True)
    status = Column(String, default="pending") # pending, extracting, correcting, done, failed
    stage = Column(String, default="pending")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processing_time_seconds = Column(Float, nullable=True)
    
    # Store JSON result as text
    result_json = Column(Text, nullable=True)


class StagingCorrection(Base):
    """
    Staging tier of the correction cache.

    Every new LLM-confirmed text correction lands here first, keyed by
    (font_hash, garbled_span).  A correction is only promoted to the trusted
    cache after CACHE_PROMOTION_THRESHOLD agreeing occurrences across
    different documents.  If two documents produce *different* corrections
    for the same key, that entry is never promoted — it stays in staging and
    keeps being resolved per-occurrence.
    """
    __tablename__ = "staging_corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    font_hash = Column(String, nullable=False, index=True)
    garbled_span = Column(String, nullable=False)
    correction = Column(String, nullable=False)
    occurrence_count = Column(Integer, default=1)
    # JSON-encoded list of document/job IDs that produced this correction
    document_ids = Column(Text, default="[]")
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("font_hash", "garbled_span", "correction",
                         name="uq_staging_font_span_correction"),
    )


class TrustedCorrection(Base):
    """
    Trusted (live) tier of the correction cache.

    Entries promoted from staging after reaching the agreeing-occurrence
    threshold.  Looked up via Aho-Corasick multi-pattern match against
    the full page text in one pass, independent of cache size.
    """
    __tablename__ = "trusted_corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    font_hash = Column(String, nullable=False, index=True)
    garbled_span = Column(String, nullable=False)
    correction = Column(String, nullable=False)
    promoted_at = Column(DateTime, default=datetime.utcnow)
    # Audit: JSON list of document IDs that contributed to promotion
    contributing_document_ids = Column(Text, default="[]")

    __table_args__ = (
        UniqueConstraint("font_hash", "garbled_span",
                         name="uq_trusted_font_span"),
    )


class ReviewFlag(Base):
    """
    Admin review queue.

    Two distinct flag types, deliberately kept visually distinguishable:
    - "text_rejection": an LLM text correction failed the edit-distance or
      confidence gate — needs OCR/text review.
    - "answer_disagreement": the LLM disagrees with the marked answer key —
      needs content/answer-key review.  NEVER auto-applied regardless of
      confidence (per spec §6).
    """
    __tablename__ = "review_flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, index=True)
    question_number = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    flag_type = Column(String, nullable=False)  # "text_rejection" | "answer_disagreement"
    original_text = Column(Text, nullable=True)
    proposed_text = Column(Text, nullable=True)
    llm_explanation = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    status = Column(String, default="pending")  # "pending" | "accepted" | "rejected"
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String, nullable=True)


class PipelineSetting(Base):
    """
    Dynamic configuration for pipeline steps, OCR engines, and feature toggles.
    Persisted in database to survive restarts and allow runtime modifications.
    """
    __tablename__ = "pipeline_config"

    key = Column(String, primary_key=True, index=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
