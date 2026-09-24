from typing import TypedDict, List, Dict, Any, Optional


class PageMetadata(TypedDict):
    """Per-page metadata preserved from extraction for downstream quality scoring."""
    page_num: int
    text: str
    font_hashes: Dict[str, str]        # {font_name: md5_hash}
    char_confidences: Dict[str, Dict]  # {font_name: {cid: resolution_source}}
    unmapped_codes: List[Dict]          # [{code: int, font: str, glyph_id: int}]
    has_text_layer: bool


class QuestionBlock(TypedDict, total=False):
    question_number: int
    stem: str
    options: Dict[str, str]
    marked_answer: Optional[str]
    explanation: Optional[str]


class ProcessedQuestion(QuestionBlock, total=False):
    validated_answer: Optional[str]
    is_valid: bool
    confidence: str
    reasoning: Optional[str]
    explanation: Optional[str]
    used_web_search: bool


class GraphState(TypedDict):
    file_path: Optional[str]
    raw_text: str
    pages: List[PageMetadata]                   # per-page extraction metadata
    # Layout-aware span stream from extract_text_from_pdf (list of LayoutElement-compatible
    # dicts: {type, text, bbox, page, font_size, is_bold, x_indent}).  Populated only for
    # PDFs with a live text layer; empty for scanned/OCR-only pages so extract_node falls
    # back to parse_questions_from_text on raw_text.
    span_blocks: List[Dict[str, Any]]
    extracted_blocks: List[QuestionBlock]
    quality_scores: Dict[int, Dict[str, Any]]   # {question_number: QualityScore dict}
    repair_log: List[Dict[str, Any]]            # log of repair actions and outcomes
    vision_calls_count: int                     # running counter for circuit breaker
    processed_questions: List[ProcessedQuestion]
    summary: Dict[str, Any]
    token_usage: Dict[str, int]
    # --- Correction pipeline fields (spec §3–§6) ---
    flagged_words: Dict[int, List[Dict]]        # {question_number: [FlaggedWord dicts]}
    correction_results: List[Dict[str, Any]]    # per-question correction outcomes
    review_flags: List[Dict[str, Any]]          # flags queued for admin review
