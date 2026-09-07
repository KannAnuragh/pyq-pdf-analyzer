from typing import TypedDict, List, Dict, Any, Optional

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
    extracted_blocks: List[QuestionBlock]
    processed_questions: List[ProcessedQuestion]
    summary: Dict[str, Any]
    token_usage: Dict[str, int]
