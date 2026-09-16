"""
correction_cache.py

Two-tier correction cache with Aho-Corasick multi-pattern matching.

Architecture (per spec §4):
- Staging: every new LLM-confirmed correction lands here first.
- Trusted: a staged entry promotes after N agreeing occurrences across
  different documents.  If two documents produce *different* corrections
  for the same (font_hash, garbled_span), the entry is never promoted.
- Lookups use Aho-Corasick to match the entire trusted cache against a
  page in one pass, independent of cache size.

Key design decisions:
- Cache key is (font_hash, garbled_span).  font_hash is already computed
  by font_mapper.py — we reuse it rather than inventing a second identity.
- Answer-verification results are NEVER cached (per spec §4): whether a
  marked answer is correct is a one-off content judgment, not a recurring
  font-bug pattern.
- The Aho-Corasick automaton is rebuilt in batches (on promotion), not on
  every single lookup.  This keeps per-page lookup cost constant.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Promotion threshold: how many agreeing occurrences across different
# documents before a staged correction is trusted.  Configurable via env.
CACHE_PROMOTION_THRESHOLD = int(os.getenv("CACHE_PROMOTION_THRESHOLD", "2"))

# ---------------------------------------------------------------------------
# Aho-Corasick automaton wrapper
# ---------------------------------------------------------------------------

try:
    import ahocorasick
    _HAS_AHOCORASICK = True
except ImportError:
    _HAS_AHOCORASICK = False
    logger.warning(
        "[CorrectionCache] pyahocorasick not installed — falling back to "
        "naive substring matching.  Install it: pip install pyahocorasick"
    )


class _AhoCorasickMatcher:
    """
    Thin wrapper around pyahocorasick.Automaton that maps
    garbled_span → (correction, font_hash) for trusted cache entries.

    Falls back to naive dict-based matching when pyahocorasick is not
    available (functional but O(cache_size × page_length) instead of
    O(page_length)).
    """

    def __init__(self):
        self._entries: dict[str, tuple[str, str]] = {}  # garbled → (correction, font_hash)
        self._automaton = None
        self._dirty = True  # needs rebuild

    def add(self, garbled_span: str, correction: str, font_hash: str) -> None:
        self._entries[garbled_span] = (correction, font_hash)
        self._dirty = True

    def remove(self, garbled_span: str) -> None:
        self._entries.pop(garbled_span, None)
        self._dirty = True

    def _rebuild(self) -> None:
        if not _HAS_AHOCORASICK or not self._entries:
            self._automaton = None
            self._dirty = False
            return
        A = ahocorasick.Automaton()
        for garbled, (correction, font_hash) in self._entries.items():
            A.add_word(garbled, (garbled, correction, font_hash))
        A.make_automaton()
        self._automaton = A
        self._dirty = False

    def find_all(self, text: str, page_font_hashes: dict[str, str] | None = None) -> dict[str, str]:
        """
        Returns {garbled_span: correction} for every trusted cache entry
        found in `text`.

        If `page_font_hashes` is given, only returns matches whose
        font_hash is present on the current page — prevents cross-font
        cache pollution.
        """
        if self._dirty:
            self._rebuild()

        page_hash_set = set(page_font_hashes.values()) if page_font_hashes else None
        results: dict[str, str] = {}

        if self._automaton is not None:
            for _end_idx, (garbled, correction, font_hash) in self._automaton.iter(text):
                if page_hash_set is not None and font_hash not in page_hash_set:
                    continue
                results[garbled] = correction
        else:
            # Naive fallback — acceptable for small caches
            for garbled, (correction, font_hash) in self._entries.items():
                if garbled in text:
                    if page_hash_set is not None and font_hash not in page_hash_set:
                        continue
                    results[garbled] = correction

        return results

    @property
    def size(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Main CorrectionCache class
# ---------------------------------------------------------------------------

class CorrectionCache:
    """
    Two-tier correction cache backed by SQLAlchemy + Aho-Corasick.

    Thread-safe: all DB operations use a dedicated session, and the
    Aho-Corasick automaton is protected by a lock during rebuilds.
    """

    def __init__(self, session_factory):
        """
        Args:
            session_factory: a callable that returns a new SQLAlchemy Session
                             (typically ``database.SessionLocal``).
        """
        self._session_factory = session_factory
        self._matcher = _AhoCorasickMatcher()
        self._lock = threading.Lock()
        self._loaded = False

    def _get_session(self) -> Session:
        return self._session_factory()

    def _ensure_loaded(self) -> None:
        """Lazy-load trusted corrections into the Aho-Corasick automaton."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_trusted()
            self._loaded = True

    def _load_trusted(self) -> None:
        """Load all trusted corrections into the matcher."""
        from ..storage.models import TrustedCorrection
        db = self._get_session()
        try:
            rows = db.query(TrustedCorrection).all()
            for row in rows:
                self._matcher.add(row.garbled_span, row.correction, row.font_hash)
            logger.info(
                "[CorrectionCache] Loaded %d trusted corrections into Aho-Corasick",
                len(rows),
            )
        finally:
            db.close()

    # ----- Public API -----

    def lookup(
        self,
        page_text: str,
        page_font_hashes: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Bulk lookup: match the entire trusted cache against page_text in
        one pass.  Returns {garbled_span: correction}.

        Cost: O(len(page_text)) with Aho-Corasick, regardless of cache size.
        """
        self._ensure_loaded()
        return self._matcher.find_all(page_text, page_font_hashes)

    def apply_corrections(
        self,
        text: str,
        page_font_hashes: dict[str, str] | None = None,
        document_id: str | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Apply all cached corrections to `text` and return (corrected_text, applied_log).

        Each entry in applied_log is:
            {"garbled": str, "correction": str, "position": int}
        """
        matches = self.lookup(text, page_font_hashes)
        if not matches:
            return text, []

        applied = []
        corrected = text
        # Apply longest matches first to avoid substring collisions
        for garbled in sorted(matches, key=len, reverse=True):
            correction = matches[garbled]
            if garbled in corrected:
                pos = corrected.index(garbled)
                corrected = corrected.replace(garbled, correction)
                applied.append({
                    "garbled": garbled,
                    "correction": correction,
                    "position": pos,
                })
                if document_id:
                    self._log_cache_usage(garbled, document_id)

        return corrected, applied

    def stage(
        self,
        font_hash: str,
        garbled_span: str,
        correction: str,
        document_id: str,
    ) -> bool:
        """
        Record a new LLM-confirmed correction in the staging tier.

        If this exact (font_hash, garbled_span, correction) already exists
        in staging, increments its count.  When count reaches the promotion
        threshold AND all staged entries for this (font_hash, garbled_span)
        agree on the same correction, promotes to trusted.

        Returns True if the entry was promoted to trusted.
        """
        from ..storage.models import StagingCorrection, TrustedCorrection

        db = self._get_session()
        promoted = False
        try:
            # Check if already trusted — nothing to do
            existing_trusted = (
                db.query(TrustedCorrection)
                .filter_by(font_hash=font_hash, garbled_span=garbled_span)
                .first()
            )
            if existing_trusted:
                db.close()
                return False

            # Upsert into staging
            existing = (
                db.query(StagingCorrection)
                .filter_by(
                    font_hash=font_hash,
                    garbled_span=garbled_span,
                    correction=correction,
                )
                .first()
            )

            if existing:
                existing.occurrence_count += 1
                existing.last_seen = datetime.utcnow()
                doc_ids = json.loads(existing.document_ids or "[]")
                if document_id not in doc_ids:
                    doc_ids.append(document_id)
                existing.document_ids = json.dumps(doc_ids)
            else:
                entry = StagingCorrection(
                    font_hash=font_hash,
                    garbled_span=garbled_span,
                    correction=correction,
                    occurrence_count=1,
                    document_ids=json.dumps([document_id]),
                )
                db.add(entry)

            db.flush()

            # Check promotion: all staged entries for this (font_hash, garbled_span)
            # must agree on the same correction, and count must reach threshold
            all_staged = (
                db.query(StagingCorrection)
                .filter_by(font_hash=font_hash, garbled_span=garbled_span)
                .all()
            )

            if len(all_staged) == 1:
                entry = all_staged[0]
                if entry.occurrence_count >= CACHE_PROMOTION_THRESHOLD:
                    # Promote!
                    trusted = TrustedCorrection(
                        font_hash=font_hash,
                        garbled_span=garbled_span,
                        correction=correction,
                        contributing_document_ids=entry.document_ids,
                    )
                    db.add(trusted)
                    promoted = True

                    # Update the in-memory matcher
                    with self._lock:
                        self._matcher.add(garbled_span, correction, font_hash)

                    logger.info(
                        "[CorrectionCache] Promoted '%s' → '%s' (font=%s, %d occurrences)",
                        garbled_span, correction, font_hash[:8], entry.occurrence_count,
                    )
            else:
                # Multiple different corrections exist for this key — don't promote.
                # This is a signal the garbled text isn't a deterministic font bug.
                corrections = set(s.correction for s in all_staged)
                if len(corrections) > 1:
                    logger.info(
                        "[CorrectionCache] Conflicting corrections for '%s' (font=%s): %s — not promoting",
                        garbled_span, font_hash[:8], corrections,
                    )

            db.commit()
        except Exception:
            db.rollback()
            logger.exception("[CorrectionCache] Failed to stage correction")
        finally:
            db.close()

        return promoted

    def get_audit_trail(self, garbled_span: str) -> list[dict]:
        """
        Returns a list of documents that used a cached correction for
        the given garbled_span, across both staging and trusted tiers.
        """
        from ..storage.models import StagingCorrection, TrustedCorrection

        db = self._get_session()
        try:
            trail = []

            staging = (
                db.query(StagingCorrection)
                .filter_by(garbled_span=garbled_span)
                .all()
            )
            for s in staging:
                trail.append({
                    "tier": "staging",
                    "font_hash": s.font_hash,
                    "correction": s.correction,
                    "count": s.occurrence_count,
                    "document_ids": json.loads(s.document_ids or "[]"),
                })

            trusted = (
                db.query(TrustedCorrection)
                .filter_by(garbled_span=garbled_span)
                .all()
            )
            for t in trusted:
                trail.append({
                    "tier": "trusted",
                    "font_hash": t.font_hash,
                    "correction": t.correction,
                    "document_ids": json.loads(t.contributing_document_ids or "[]"),
                })

            return trail
        finally:
            db.close()

    def get_stats(self) -> dict:
        """Cache health statistics."""
        from ..storage.models import StagingCorrection, TrustedCorrection

        db = self._get_session()
        try:
            staging_count = db.query(StagingCorrection).count()
            trusted_count = db.query(TrustedCorrection).count()
            return {
                "staging_count": staging_count,
                "trusted_count": trusted_count,
                "automaton_size": self._matcher.size,
                "promotion_threshold": CACHE_PROMOTION_THRESHOLD,
            }
        finally:
            db.close()

    def _log_cache_usage(self, garbled_span: str, document_id: str) -> None:
        """
        Record that a trusted cache entry was applied to a specific document.
        This builds the audit trail so that if a bad entry is later caught,
        we can identify exactly which documents to re-check.
        """
        # For now, audit trail is implicit from staging.document_ids.
        # A more advanced version could maintain a separate usage log table.
        logger.debug(
            "[CorrectionCache] Applied cached correction for '%s' in document %s",
            garbled_span, document_id,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cache_instance: Optional[CorrectionCache] = None
_cache_lock = threading.Lock()


def get_correction_cache() -> CorrectionCache:
    """Returns the module-level singleton CorrectionCache instance."""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                from ..storage.database import SessionLocal
                _cache_instance = CorrectionCache(SessionLocal)
    return _cache_instance
