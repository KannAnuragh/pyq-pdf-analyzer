"""
word_flag_queue.py

Persistence layer for malayalam_validator.py flags, built on the SQLAlchemy/
SQLite stack your project already uses for job history (per your existing
models.py — adjust the Base/session imports below to match your actual
module paths, they're guessed at [Guessing] since I don't have your source).

One row per flagged word per job. This is what your review UI and your
targeted-crop OCR fallback both read from.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Float, DateTime, Enum, JSON, ForeignKey
from sqlalchemy.orm import relationship

# [Guessing] Adjust this import to wherever your existing Base/engine lives —
# e.g. `from db.base import Base` or `from models import Base`.
from db.base import Base  # noqa: replace with your actual import


class FlagStatus(str, enum.Enum):
    PENDING = "pending"            # not yet acted on
    AUTO_REPAIRED = "auto_repaired"  # repaired_missing_space, applied automatically
    OCR_SUGGESTED = "ocr_suggested"  # crop-OCR ran, has a candidate correction
    RESOLVED = "resolved"          # human or judge-agent confirmed a final value
    IGNORED = "ignored"            # reviewed and judged a false positive


class FlaggedWord(Base):
    __tablename__ = "flagged_words"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, index=True)  # [Guessing] your jobs table's PK name

    page_number = Column(Integer, nullable=True)
    char_offset = Column(Integer, nullable=False)
    original_text = Column(String, nullable=False)
    reason = Column(String, nullable=False)          # from Flag.reason
    suggestion = Column(String, nullable=True)        # from Flag.suggestion (segmenter repair)
    ocr_candidate = Column(String, nullable=True)      # filled in by ocr_word_fallback
    ocr_confidence = Column(Float, nullable=True)

    # bbox of the token on the page, [x0, y0, x1, y1] in PDF points — needed
    # to crop the region for targeted OCR. See integration note on why this
    # has to be threaded through from router.py's span data, not recovered
    # after the text is already flattened to a string.
    bbox = Column(JSON, nullable=True)

    status = Column(Enum(FlagStatus), default=FlagStatus.PENDING, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    job = relationship("Job", back_populates="flagged_words")  # [Guessing] add
    # `flagged_words = relationship("FlaggedWord", back_populates="job")`
    # to your existing Job model


# --------------------------------------------------------------------------- #
# Helpers called from the ingestion pipeline
# --------------------------------------------------------------------------- #

def persist_flags(session, job_id: str, page_number: int, flags: list, bbox_lookup=None) -> list[FlaggedWord]:
    """
    flags: the `ValidationResult.flags` list from malayalam_validator.validate_and_repair().
    bbox_lookup: optional callable(char_offset:int) -> [x0,y0,x1,y1] | None,
    for mapping a flagged token back to its position on the page. If you
    don't have per-token bbox data wired through yet, pass None — flags
    still get recorded, you just can't do targeted-crop OCR on them until
    bbox is available (see integration note 2).
    """
    rows = []
    for f in flags:
        bbox = bbox_lookup(f.char_offset) if bbox_lookup else None
        status = FlagStatus.AUTO_REPAIRED if f.reason == "repaired_missing_space" else FlagStatus.PENDING
        row = FlaggedWord(
            job_id=job_id,
            page_number=page_number,
            char_offset=f.char_offset,
            original_text=f.span,
            reason=f.reason,
            suggestion=f.suggestion,
            bbox=bbox,
            status=status,
        )
        session.add(row)
        rows.append(row)
    session.commit()
    return rows


def pending_ocr_candidates(session, job_id: str) -> list[FlaggedWord]:
    """Flags that need a targeted-crop OCR pass: has a bbox, no OCR run yet."""
    return (
        session.query(FlaggedWord)
        .filter(
            FlaggedWord.job_id == job_id,
            FlaggedWord.status == FlagStatus.PENDING,
            FlaggedWord.bbox.isnot(None),
        )
        .all()
    )


def needs_human_review(session, job_id: str) -> list[FlaggedWord]:
    """Flags with no bbox (can't auto-OCR) or where OCR confidence was low —
    these are what your review UI should actually show a human."""
    return (
        session.query(FlaggedWord)
        .filter(
            FlaggedWord.job_id == job_id,
            FlaggedWord.status.in_([FlagStatus.PENDING, FlagStatus.OCR_SUGGESTED]),
        )
        .all()
    )
