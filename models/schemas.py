from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, List, Any
from datetime import datetime

class QuestionResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    
    question_number: int
    stem: str
    options: Dict[str, str]
    marked_correct_answer: Optional[str] = None
    marked_answer: Optional[str] = None
    validated_answer: Optional[str] = None
    is_valid: bool = False
    confidence: str = "low"
    reasoning: Optional[str] = None
    explanation: Optional[str] = None
    used_web_search: bool = False
    page_number: Optional[int] = None
    pipeline_steps: Optional[Dict[str, Any]] = None
    token_usage: Optional["TokenUsage"] = None

class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

class Summary(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    total: int = 0
    validated: int = 0
    flagged: int = 0
    unvalidated: int = 0
    gemini_enabled: bool = True
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

class PDFResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    source_file: str
    processed_at: datetime
    questions: List[QuestionResult] = Field(default_factory=list)
    summary: Summary = Field(default_factory=Summary)

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    created_at: datetime
    has_file: bool = True
    question_count: Optional[int] = None

class JobResultResponse(BaseModel):
    job_id: str
    result: Optional[PDFResult]


# ---------------------------------------------------------------------------
# Review flag API models (spec §6 — admin review queue)
# ---------------------------------------------------------------------------

class ReviewFlagResponse(BaseModel):
    """Single review flag for the admin queue."""
    id: int
    job_id: str
    question_number: int
    page_number: Optional[int] = None
    flag_type: str  # "text_rejection" | "answer_disagreement"
    original_text: Optional[str] = None
    proposed_text: Optional[str] = None
    llm_explanation: Optional[str] = None
    confidence: Optional[float] = None
    status: str = "pending"
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class ReviewFlagListResponse(BaseModel):
    """Paginated list of review flags."""
    flags: List[ReviewFlagResponse]
    total: int


class ReviewFlagResolveRequest(BaseModel):
    """Request body for resolving a review flag."""
    action: str  # "accept" | "reject"
    resolved_by: str = "admin"


class ReviewFlagStatsResponse(BaseModel):
    """Aggregate review flag statistics."""
    total_pending: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    text_rejections_pending: int = 0
    answer_disagreements_pending: int = 0


class CacheStatsResponse(BaseModel):
    """Correction cache health statistics."""
    staging_count: int = 0
    trusted_count: int = 0
    automaton_size: int = 0
    promotion_threshold: int = 2

