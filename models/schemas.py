from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, List
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
    used_web_search: bool = False

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

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

class JobResultResponse(BaseModel):
    job_id: str
    result: Optional[PDFResult]
