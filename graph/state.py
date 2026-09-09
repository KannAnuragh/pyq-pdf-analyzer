from typing import TypedDict, List, Dict, Any, Optional


class PageMetadata(TypedDict):
    """Per-page metadata preserved from extraction for downstream quality scoring."""
    page_num: int
    text: str
    font_hashes: Dict[str, str]        # {font_name: md5_hash}
    unmapped_codes: List[Dict]          # [{code: int, font: str, glyph_id: int}]
    has_text_layer: bool


class QuestionBlock(TypedDict):
    question_number: int
    stem: str
    options: Dict[str, str]
    marked_answer: Optional[str]


class ProcessedQuestion(QuestionBlock):
    validated_answer: Optional[str]
    is_valid: bool
    confidence: str
    reasoning: Optional[str]
    used_web_search: bool


class GraphState(TypedDict):
    file_path: Optional[str]
    raw_text: str
    pages: List[PageMetadata]                   # per-page extraction metadata
    extracted_blocks: List[QuestionBlock]
    quality_scores: Dict[int, Dict[str, Any]]   # {question_number: QualityScore dict}
    repair_log: List[Dict[str, Any]]            # log of repair actions and outcomes
    vision_calls_count: int                     # running counter for circuit breaker
    processed_questions: List[ProcessedQuestion]
    summary: Dict[str, Any]
    token_usage: Dict[str, int]
