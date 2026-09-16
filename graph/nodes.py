import json
import os
import re
import traceback
from typing import Optional, Dict, List, Tuple, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
from .state import GraphState, QuestionBlock, ProcessedQuestion
from ..prompts.templates import EXTRACT_PROMPT, SOLVE_PROMPT

def get_gemini_status() -> dict:
    """Returns detailed status of Gemini integration."""
    enable = os.getenv("ENABLE_GEMINI", "false").lower() in ("true", "1", "yes")
    key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not enable:
        return {"enabled": False, "reason": "Disabled in .env (ENABLE_GEMINI=false)"}
    if not key or key.startswith("your_"):
        return {"enabled": False, "reason": "GOOGLE_API_KEY is not set"}
    return {"enabled": True, "reason": "Active"}


def is_gemini_enabled() -> bool:
    """Checks if Gemini API integration is explicitly enabled and API key is available."""
    return get_gemini_status()["enabled"]


def get_groq_status() -> dict:
    """Returns detailed status of Groq integration."""
    enable = os.getenv("ENABLE_GROQ", "false").lower() in ("true", "1", "yes")
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not enable:
        return {"enabled": False, "reason": "Disabled in .env (ENABLE_GROQ=false)"}
    if not key or key.startswith("your_"):
        return {"enabled": False, "reason": "GROQ_API_KEY is not set"}
    return {"enabled": True, "reason": "Active"}


def is_groq_enabled() -> bool:
    """Checks if Groq API integration is explicitly enabled and API key is available."""
    return get_groq_status()["enabled"]


def get_llm_status() -> dict:
    """Returns status of the active LLM provider (Groq or Gemini)."""
    groq_st = get_groq_status()
    if groq_st["enabled"]:
        return {"enabled": True, "provider": "groq", "reason": "Active (Groq)"}
    gemini_st = get_gemini_status()
    if gemini_st["enabled"]:
        return {"enabled": True, "provider": "gemini", "reason": "Active (Gemini)"}
    return {
        "enabled": False,
        "provider": "none",
        "reason": f"No LLM active. Groq: {groq_st['reason']}; Gemini: {gemini_st['reason']}"
    }


def is_llm_enabled() -> bool:
    return get_llm_status()["enabled"]


def is_answer_validation_enabled() -> bool:
    """Checks whether AI answer solving & verification is enabled via runtime configuration or env."""
    try:
        from ..storage.config_manager import get_setting
        return bool(get_setting("validation_enabled", True))
    except Exception:
        val = os.getenv("ENABLE_ANSWER_VALIDATION", os.getenv("ENABLE_VALIDATION", "true"))
        return val.strip().lower() in ("true", "1", "yes")


def get_extract_llm(provider: Optional[str] = None):
    """Returns an extraction LLM from Groq (if enabled) or Gemini."""
    if not provider:
        try:
            from ..storage.config_manager import get_setting
            provider = get_setting("validation_provider")
        except Exception:
            pass

    if not provider or provider not in ("groq", "gemini"):
        llm_st = get_llm_status()
        provider = llm_st.get("provider", "none")
    
    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
            model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
            key = (os.getenv("GROQ_API_KEY") or "").strip()
            return ChatGroq(model=model_name, temperature=0.1, api_key=key)
        except Exception as e:
            print(f"Warning: Failed to initialize Groq extract_llm: {e}", flush=True)
            return None
    elif provider == "gemini":
        try:
            model_name = os.getenv("GEMINI_EXTRACT_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
            key = (os.getenv("GOOGLE_API_KEY") or "").strip()
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,
                google_api_key=key
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Gemini extract_llm: {e}", flush=True)
            return None
    return None


def get_solve_llm(provider: Optional[str] = None):
    """Returns a solving & validation LLM from Groq (if enabled) or Gemini."""
    if not provider:
        try:
            from ..storage.config_manager import get_setting
            provider = get_setting("validation_provider")
        except Exception:
            pass

    if not provider or provider not in ("groq", "gemini"):
        llm_st = get_llm_status()
        provider = llm_st.get("provider", "none")

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
            model_name = os.getenv("GROQ_SOLVE_MODEL", os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
            key = (os.getenv("GROQ_API_KEY") or "").strip()
            return ChatGroq(model=model_name, temperature=0.1, api_key=key)
        except Exception as e:
            print(f"Warning: Failed to initialize Groq solve_llm: {e}", flush=True)
            return None
    elif provider == "gemini":
        try:
            model_name = os.getenv("GEMINI_SOLVE_MODEL", "gemini-3.1-pro-preview")
            key = (os.getenv("GOOGLE_API_KEY") or "").strip()
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,
                google_api_key=key
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Gemini solve_llm: {e}", flush=True)
            return None
    return None

def _extract_text_content(content) -> str:
    """Safely extracts a string from AIMessage.content (which can be str or list of dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)

def _clean_json_output(content) -> str:
    """Strips Markdown backticks and extraneous text around JSON."""
    raw = _extract_text_content(content).strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    return raw.strip()



MULTIMODAL_EXTRACT_PROMPT = """
Extract all questions from this uploaded PDF document into a structured JSON array.

CRITICAL INSTRUCTIONS:
1. Preserve all mathematical symbols, formulas, and Greek letters as standard inline LaTeX ($...$).
   - For example: chi-squared must be written as $\\chi^2$, t-test as 't' test or $t$-test, z-test as $z$-test, alpha as $\\alpha$, beta as $\\beta$, sigma as $\\sigma$, etc.
   - Any inline symbols, equations, superscripts, and subscripts must be output as valid LaTeX math ($...$).
2. Extract all options cleanly into 'options': {"A": "...", "B": "...", "C": "...", "D": "..."}.
   - NEVER leave options (A:-, B:-, C:-, D:-) inside the question 'stem'.
3. Extract the given 'Correct Answer' into 'marked_answer' (e.g. "A", "B", "C", "D") if present in the document.
4. If a question contains sub-items or statements, format each on its own line in 'stem'.
   - For matching questions (List I / List II), keep each paired row on the same line separated by a tab (\t).
   - Within options, keep code sequences like '(a) (b) (c) (d) (e)' together on a single line; never split them into single-letter lines.

Output ONLY a valid JSON array matching this schema:
[
  {
    "question_number": 1,
    "stem": "Question stem text here\\na. $\\\\chi^2$ test\\nb. 't' test",
    "options": {
      "A": "Option A text",
      "B": "Option B text",
      "C": "Option C text",
      "D": "Option D text"
    },
    "marked_answer": "B"
  }
]
"""

def parse_trailing_meta(text: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    Parses trailing metadata from text (e.g. at the end of an option or question block).
    Detects marked answer (English 'Answer: B', 'Ans:- B', 'Correct Answer:- Option-C', Malayalam 'ഉത്തരം: B', etc.)
    and solution/explanation ('Solution:', 'Explanation:', 'Description:', 'വിശദീകരണം:', 'പരിഹാരം:', etc.).
    Returns (clean_text, marked_answer, explanation).
    """
    if not text:
        return "", None, None

    clean_text = text.strip()
    marked_answer = None
    explanation = None

    ans_pat = re.compile(
        r"(?:^|\n|\s+)(?:Answer|Ans(?:wer)?|Correct(?:\s*Answer)?|ഉത്തരം)\s*[:\-]*\s*(?:is\s*)?(?:Option\s*[:\-]*\s*)?(?:\()?([A-D])(?:\))?\b",
        re.IGNORECASE
    )
    sol_pat = re.compile(
        r"(?:^|\n|\s+)(?:Solution|Explanation|Description|Sol|Exp|Hint|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*[:\.\-]+\s*",
        re.IGNORECASE
    )

    ans_m = ans_pat.search(clean_text)
    sol_m = sol_pat.search(clean_text)

    if ans_m and sol_m:
        if ans_m.start() < sol_m.start():
            marked_answer = ans_m.group(1).upper()
            explanation = clean_text[sol_m.end():].strip()
            clean_text = clean_text[:ans_m.start()].strip()
        else:
            marked_answer = ans_m.group(1).upper()
            explanation = (clean_text[sol_m.end():ans_m.start()] + "\n" + clean_text[ans_m.end():]).strip()
            clean_text = clean_text[:sol_m.start()].strip()
    elif ans_m:
        marked_answer = ans_m.group(1).upper()
        trailing = clean_text[ans_m.end():].strip()
        trailing_sol_m = sol_pat.search(trailing)
        if trailing_sol_m:
            explanation = trailing[trailing_sol_m.end():].strip()
        elif trailing and (trailing.lower().startswith("because") or trailing.lower().startswith("since")):
            explanation = trailing
        clean_text = clean_text[:ans_m.start()].strip()
    elif sol_m:
        explanation = clean_text[sol_m.end():].strip()
        clean_text = clean_text[:sol_m.start()].strip()
        ans_in_sol = re.search(r"(?:Option\s*[:\-]*\s*\(?([A-D])\)?\s*is\s*correct|\(?([A-D])\)?\s*is\s*(?:the\s*)?correct\s*(?:option|answer))", explanation, re.IGNORECASE)
        if ans_in_sol:
            marked_answer = (ans_in_sol.group(1) or ans_in_sol.group(2)).upper()

    return clean_text, marked_answer, explanation


def separate_stem_and_options(content: str) -> tuple[str, dict[str, str], Optional[str], Optional[str]]:
    """
    Given raw question content, reliably separates question stem from multiple-choice options (A, B, C, D).
    Guarantees:
    1. Heals mangled option markers (e.g. 1\nB:- turned into \frac{1}{B}:-).
    2. Finds increasing option sequence (A, B, C, D or A, C, D or B, C, D).
    3. Emergency line fallback if options are at the bottom of the stem.
    4. Strips and extracts marked answers ('Correct Answer:- Option-C', 'Ans:- B', 'ഉത്തരം: C').
    5. Extracts solution/description ('Solution:', 'Explanation:', 'വിശദീകരണം:') if present in the document.
    Returns (stem, options, marked_answer, explanation).
    """
    if not content:
        return "", {}, None, None

    # Heal any accidental math conversions of option markers (e.g. 1\nB:- turned into \frac{1}{B}:-)
    content = re.sub(
        r'(?:\$?\{\\frac\{1\}\{([A-D])\}\}?|\$?\\\s*frac\s*\{\s*1\s*\}\s*\{\s*([A-D])\s*\}\$?|1\s*/\s*([A-D]))(\s*[:-]+|\s*\.|\s*\))',
        lambda m: f"1\n{m.group(1) or m.group(2) or m.group(3)}{m.group(4)}",
        content
    )

    opt_pattern = re.compile(r'(?:^|\n|\s+)(?:\()?([A-D])(?:\)|\.|:-)\s*')
    matches = list(opt_pattern.finditer(content))
    
    best_sequence = []
    for i, m in enumerate(matches):
        curr_seq = [m]
        curr_letter = m.group(1)
        for next_m in matches[i+1:]:
            next_letter = next_m.group(1)
            if next_letter > curr_letter:
                curr_seq.append(next_m)
                curr_letter = next_letter
        if len(curr_seq) > len(best_sequence):
            best_sequence = curr_seq

    stem = content.strip()
    options = {}
    marked_answer = None
    explanation = None

    if len(best_sequence) >= 2:
        stem = content[:best_sequence[0].start()].strip()
        for idx, m in enumerate(best_sequence):
            letter = m.group(1)
            start_pos = m.end()
            end_pos = best_sequence[idx+1].start() if idx < len(best_sequence) - 1 else len(content)
            opt_text = content[start_pos:end_pos].strip()
            
            clean_opt, ans, exp = parse_trailing_meta(opt_text)
            if ans and not marked_answer:
                marked_answer = ans
            if exp and not explanation:
                explanation = exp
            options[letter] = clean_opt

    # Fallback 1: Flexible separator matching
    if len(options) < 2:
        fallback_pattern = re.compile(r'(?:^|\s)(?:\()?([A-D])(?:\)|\.|:-)\s*')
        fb_matches = list(fallback_pattern.finditer(content))
        if fb_matches:
            last_positions = {}
            for m in fb_matches:
                last_positions[m.group(1)] = m
            if len(last_positions) >= 2:
                true_option_matches = sorted(last_positions.values(), key=lambda m: m.start())
                stem = content[:true_option_matches[0].start()].strip()
                for i, match in enumerate(true_option_matches):
                    opt_letter = match.group(1)
                    start_idx = match.end()
                    end_idx = true_option_matches[i+1].start() if i + 1 < len(true_option_matches) else len(content)
                    opt_text = content[start_idx:end_idx].strip()
                    clean_opt, ans, exp = parse_trailing_meta(opt_text)
                    if ans and not marked_answer:
                        marked_answer = ans
                    if exp and not explanation:
                        explanation = exp
                    options[opt_letter] = clean_opt

    # Fallback 2: Line-based emergency extraction if options are still trapped in stem
    if len(options) < 2:
        lines = stem.split('\n')
        opt_line_pattern = re.compile(r'^\s*(?:\()?([A-D])(?:\)|\.|:-)\s*(.*)', re.IGNORECASE)
        first_opt_idx = -1
        found_opts = {}
        for idx, line in enumerate(lines):
            m_line = opt_line_pattern.match(line)
            if m_line:
                if first_opt_idx == -1:
                    first_opt_idx = idx
                opt_t = m_line.group(2).strip()
                clean_opt, ans, exp = parse_trailing_meta(opt_t)
                if ans and not marked_answer:
                    marked_answer = ans
                if exp and not explanation:
                    explanation = exp
                found_opts[m_line.group(1).upper()] = clean_opt
        if len(found_opts) >= 2:
            options = found_opts
            stem = '\n'.join(lines[:first_opt_idx]).strip()
    
    # Check if marked answer or explanation was in the stem
    if not marked_answer or not explanation:
        clean_stem, s_ans, s_exp = parse_trailing_meta(stem)
        if s_ans and not marked_answer:
            marked_answer = s_ans
            stem = clean_stem
        if s_exp and not explanation:
            explanation = s_exp
            stem = clean_stem

    return stem, options, marked_answer, explanation


def parse_questions_from_text(raw_text: str, token_tracker: dict = None) -> list[dict]:
    """Extracts question blocks from raw text using regex with LLM fallback."""
    extracted = []
    
    # 1. Determine if the document explicitly uses 'Question' or 'Q' markers
    explicit_matches = list(re.finditer(r'(?i)(?:^|\n|\s+)\s*(?:Question\s*|Q\.?\s*)(\d+)\s*(?::\-?|[\.\):-])\s*', raw_text))
    if len(explicit_matches) >= 3:
        # Document uses explicit Question/Q markers (e.g. 'Question 1:-', 'Question20:-', 'Q.1:')
        q_prefix_pattern = r'(?:Question\s*|Q\.?\s*)\d+\s*(?::\-?|[\.\):-])\s*'
        q_extract_pattern = r'^(?:Question\s*|Q\.?\s*)(\d+)\s*(?::\-?|[\.\):-])\s*(.*)'
        blocks = re.split(fr'(?:\n|\s+|^)(?={q_prefix_pattern})', raw_text, flags=re.MULTILINE | re.IGNORECASE)
    else:
        # Fallback for documents with simple numeric numbering (e.g. '1. ', '2. ', '1) ')
        q_prefix_pattern = r'\d+\s*(?:[\.\)]|:-)\s+'
        q_extract_pattern = r'^(\d+)\s*(?:[\.\)]|:-)\s+(.*)'
        blocks = re.split(fr'\n(?={q_prefix_pattern})|^(?={q_prefix_pattern})', raw_text, flags=re.MULTILINE | re.IGNORECASE)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # Match question number
        q_match = re.match(q_extract_pattern, block, flags=re.DOTALL | re.IGNORECASE)
        if not q_match:
            continue
            
        q_num = int(q_match.group(1))
        content = q_match.group(2)
        
        stem, options, marked_answer, explanation = separate_stem_and_options(content)
        
        extracted.append({
            "question_number": q_num,
            "stem": stem,
            "options": options,
            "marked_answer": marked_answer,
            "explanation": explanation
        })
        
    extracted.sort(key=lambda x: x["question_number"])
    print(f"Regex parser extracted {len(extracted)} questions.", flush=True)
    
    avg_length = sum(len(re.sub(r'<[^>]+>', '', b["stem"])) for b in extracted) / max(1, len(extracted))
    
    # Fallback to LLM if regex fails (extracts very few questions or creates huge malformed blocks)
    if len(extracted) == 0 or (len(raw_text) > 1000 and len(extracted) < 2) or avg_length > 1500:
        llm = get_extract_llm()
        if llm:
            print("Regex parser extracted too few questions or malformed blocks. Falling back to Gemini extraction...", flush=True)
            prompt = EXTRACT_PROMPT.replace("{text}", raw_text)
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                
                # Track fallback tokens
                p_tokens = response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
                c_tokens = response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
                
                if token_tracker is not None:
                    token_tracker["prompt_tokens"] = token_tracker.get("prompt_tokens", 0) + p_tokens
                    token_tracker["completion_tokens"] = token_tracker.get("completion_tokens", 0) + c_tokens
                    token_tracker["total_tokens"] = token_tracker.get("total_tokens", 0) + (p_tokens + c_tokens)
                
                cleaned = _clean_json_output(response.content)
                extracted = json.loads(cleaned)
                if isinstance(extracted, dict) and "questions" in extracted:
                    extracted = extracted["questions"]
                elif not isinstance(extracted, list):
                    extracted = [extracted]
                print(f"Successfully extracted {len(extracted)} questions using Gemini fallback.", flush=True)
            except Exception as e:
                print(f"Extraction fallback failed with Gemini: {e}", flush=True)
                traceback.print_exc()
        else:
            print("Gemini extraction is disabled or API key not set. Using regex extracted questions.", flush=True)

    return extracted or []


def extract_node(state: GraphState) -> GraphState:
    """Extracts question blocks from raw text using a lightweight Regex parser."""
    raw_text = state.get("raw_text", "")
    if "token_usage" not in state:
        state["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    extracted = parse_questions_from_text(raw_text, state["token_usage"])
    return {"extracted_blocks": extracted, "token_usage": state.get("token_usage", {})}


def quality_score_node(state: GraphState) -> GraphState:
    """
    Scores every extracted question at $0 cost (pure regex/string analysis).
    Replaces the old "5 random questions" sampling approach.

    Classifies each question as: ok | font_issue | structural_issue | unreadable
    and attaches the diagnosis to state for repair_node to consume.
    """
    from ..ingest.quality_check import score_question_text, diagnose_failure_type

    blocks = state.get("extracted_blocks", [])
    pages = state.get("pages", [])
    quality_scores = {}
    needs_repair = []

    # Build a lookup of unmapped codes by page
    page_unmapped = {}
    for p in pages:
        page_unmapped[p.get("page_num", 0)] = p.get("unmapped_codes", [])

    # Determine if document has Malayalam content
    raw_text = state.get("raw_text", "")
    has_malayalam = bool(re.search(r'[\u0D00-\u0D7F]', raw_text))

    # Gather all validation flags across pages
    all_page_flags = []
    for p in pages:
        all_page_flags.extend(p.get("validation_flags", []))

    for b in blocks:
        q_num = b.get("question_number", 0)
        stem = b.get("stem", "")
        options = b.get("options", {})

        # Combine stem + options for scoring
        full_text = stem + "\n" + "\n".join(f"({k}) {v}" for k, v in options.items())

        score = score_question_text(full_text, has_malayalam_expected=has_malayalam, page_flags=all_page_flags)
        score_dict = {
            "coverage_pct": score.coverage_pct,
            "unmapped_glyph_count": score.unmapped_glyph_count,
            "suspicious_gap_count": score.suspicious_gap_count,
            "structural_anomalies": score.structural_anomalies,
            "validation_flags": score.validation_flags,
            "status": score.status,
        }
        quality_scores[q_num] = score_dict

        if score.status != "ok":
            needs_repair.append(q_num)

    ok_count = sum(1 for s in quality_scores.values() if s["status"] == "ok")
    issue_count = len(quality_scores) - ok_count
    print(f"[Quality Check] Scored {len(quality_scores)} questions: "
          f"{ok_count} ok, {issue_count} need repair.", flush=True)

    return {
        "quality_scores": quality_scores,
    }


def repair_node(state: GraphState) -> GraphState:
    """
    Tiered repair node — applies the cheapest viable fix for each failing question.

    Tiers:
      0   — structural bugs (reordering, band merge): log only, $0
      1   — unmapped glyphs / unknown font: batch glyph repair via single Vision call
      1.5 — equation crop failure: re-detect/re-render, $0
      2   — scan quality: targeted crop OCR (last resort, circuit-breaker protected)

    Cost savings vs. old approach:
      - Old: full-page Vision OCR on any sanity failure (~5000 tokens/page)
      - New: typically 0-1 Vision calls total per document (~200-500 tokens for glyph grid)
    """
    from ..ingest.quality_check import score_question_text, diagnose_failure_type
    from ..ingest.font_analyzer import repair_unmapped_glyphs
    from ..ingest.font_installer import validate_and_commit_font_map
    from ..ingest.text_cleaner import auto_fix_deterministic

    quality_scores = state.get("quality_scores", {})
    blocks = state.get("extracted_blocks", [])
    pages = state.get("pages", [])
    repair_log = state.get("repair_log", [])
    vision_calls = state.get("vision_calls_count", 0)
    token_usage = state.get("token_usage", {})

    max_ocr = int(os.getenv("MAX_TARGETED_OCR_PER_DOCUMENT", "10"))

    # Check if repair is enabled via configuration
    try:
        from ..storage.config_manager import get_setting
        if not get_setting("repair_enabled", True):
            print("[Repair] Question quality repair disabled by configuration. Skipping.", flush=True)
            return {"repair_log": repair_log, "vision_calls_count": vision_calls}
    except Exception:
        pass

    # Find questions needing repair
    failing_q_nums = [qn for qn, s in quality_scores.items() if s.get("status") != "ok"]

    if not failing_q_nums:
        print("[Repair] No questions need repair.", flush=True)
        return {"repair_log": repair_log, "vision_calls_count": vision_calls}

    # -----------------------------------------------------------------------
    # AUTOMATIC WHOLE PDF OCR ESCALATION
    # Check if 3-5 consecutive questions are unreadable or severely corrupted
    # -----------------------------------------------------------------------
    try:
        from ..storage.config_manager import get_setting
        streak_threshold = int(get_setting("auto_whole_pdf_ocr_threshold", 3))
    except Exception:
        streak_threshold = int(os.getenv("AUTO_WHOLE_PDF_OCR_THRESHOLD", "3"))

    def _is_unreadable_question(qn: int) -> bool:
        sc = quality_scores.get(qn, {})
        st = sc.get("status", "")
        cov = sc.get("coverage_pct", 1.0)
        unmapped = sc.get("unmapped_glyph_count", 0)
        blk = next((b for b in blocks if b.get("question_number") == qn), None)
        return (
            st in ("unreadable", "scan_quality")
            or (st == "font_issue" and cov < 0.60)
            or unmapped >= 10
            or (blk is not None and not blk.get("stem", "").strip())
        )

    sorted_q_nums = sorted([b.get("question_number", 0) for b in blocks])
    longest_unreadable_streak = []
    curr_unreadable_streak = []
    for qn in sorted_q_nums:
        if _is_unreadable_question(qn):
            curr_unreadable_streak.append(qn)
            if len(curr_unreadable_streak) > len(longest_unreadable_streak):
                longest_unreadable_streak = list(curr_unreadable_streak)
        else:
            curr_unreadable_streak = []

    if streak_threshold > 0 and len(longest_unreadable_streak) >= streak_threshold:
        file_path = state.get("file_path", "")
        if file_path and os.path.exists(file_path):
            print(
                f"[Repair] Detected {len(longest_unreadable_streak)} consecutive unreadable questions "
                f"(Q{longest_unreadable_streak[0]}-Q{longest_unreadable_streak[-1]}). "
                f"Escalating to full PDF OCR scan...",
                flush=True
            )
            from ..ingest.router import ocr_whole_pdf
            try:
                ocr_engine = get_setting("auto_whole_pdf_ocr_engine", "auto")
            except Exception:
                ocr_engine = os.getenv("AUTO_WHOLE_PDF_OCR_ENGINE", "auto")

            ocr_res = ocr_whole_pdf(file_path, engine=ocr_engine)
            recovered_blocks = parse_questions_from_text(ocr_res["full_text"])
            if recovered_blocks and len(recovered_blocks) >= max(1, int(len(blocks) * 0.4)):
                print(
                    f"[Repair] Full PDF OCR successfully recovered {len(recovered_blocks)} questions. "
                    f"Replacing corrupted document blocks.",
                    flush=True
                )
                blocks = recovered_blocks
                state["extracted_blocks"] = blocks
                state["raw_text"] = ocr_res["full_text"]
                state["pages"] = ocr_res["pages"]

                p_tok = ocr_res.get("prompt_tokens", 0)
                c_tok = ocr_res.get("completion_tokens", 0)
                if "token_usage" not in state:
                    state["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                state["token_usage"]["prompt_tokens"] += p_tok
                state["token_usage"]["completion_tokens"] += c_tok
                state["token_usage"]["total_tokens"] += (p_tok + c_tok)

                repair_log.append({
                    "action": "ocr_whole_pdf_escalation",
                    "result": f"rescanned_{len(recovered_blocks)}_questions",
                    "unreadable_streak": longest_unreadable_streak,
                })

                return {
                    "extracted_blocks": blocks,
                    "raw_text": ocr_res["full_text"],
                    "pages": ocr_res["pages"],
                    "quality_scores": {},
                    "repair_log": repair_log,
                    "vision_calls_count": vision_calls + 1,
                    "token_usage": state["token_usage"],
                }

    print(f"[Repair] {len(failing_q_nums)} questions need repair. Diagnosing...", flush=True)

    # Build page data lookup
    page_unmapped = {}
    page_font_hashes = {}
    for p in pages:
        pn = p.get("page_num", 0)
        page_unmapped[pn] = p.get("unmapped_codes", [])
        page_font_hashes[pn] = p.get("font_hashes", {})

    # Collect all unmapped codes by font hash for batched glyph repair
    font_unmapped_batches = {}  # {font_hash: {"codes": [...], "font_name": str, "question_nums": [...]}}

    # Classify each failing question
    raw_text = state.get("raw_text", "")
    has_malayalam = bool(re.search(r'[ഀ-ൿ]', raw_text))

    for q_num in failing_q_nums:
        block = next((b for b in blocks if b.get("question_number") == q_num), None)
        if not block:
            continue

        score = quality_scores.get(q_num, {})
        stem = block.get("stem", "")
        options = block.get("options", {})
        full_text = stem + "\n" + "\n".join(
            f"({k}) {v}" for k, v in options.items()
        )

        # Gather unmapped codes across all pages
        all_unmapped = []
        for codes in page_unmapped.values():
            all_unmapped.extend(codes)
            
        # Gather validation flags across all pages
        all_page_flags = []
        for p in pages:
            all_page_flags.extend(p.get("validation_flags", []))

        failure_type = diagnose_failure_type(
            full_text, [], {},
            unmapped_codes=all_unmapped if score.get("status") == "font_issue" else None,
            has_text_layer=any(p.get("has_text_layer", True) for p in pages),
            validation_flags=score.get("validation_flags", []) if score.get("status") == "word_level_corruption" else None,
        )

        # -------------------------------------------------------------
        # TIER 0: Deterministic Rule-Based Auto-Fix ($0 cost, 0 Vision calls)
        # -------------------------------------------------------------
        # Fixes:
        #   - Extra / double spaces, space before punctuation, spaces around combining signs
        #   - Orphaned matras (reordered per convert_visual_to_logical rules to attach to nearest valid consonant)
        #   - Stray zero-width / control characters (ZWSP, BOM, dotted circle, non-virama ZWNJ/ZWJ)
        #   - Duplicate-character leaks from band merging (3+ repeated consonants, duplicate subjoined signs)
        #
        # Safety Guarantees:
        #   - Transformations are rule-derived, NOT inferred ($0 cost, no Vision call, no validation gate needed).
        #   - NEVER performs ambiguous single-glyph substitution ("this letter looks wrong"):
        #     Any isolated Latin characters, unmapped symbols, numbers, or formula variables are strictly
        #     preserved as-is so they remain visibly flagged for reviewers/Vision rather than silently
        #     guessing plausible wrong letters that could alter exam answers.
        # -------------------------------------------------------------
        has_structural_anomalies = bool(
            score.get("structural_anomalies") or
            score.get("suspicious_gap_count", 0) > 0 or
            failure_type in ("reordering_bug", "band_merge_bug") or
            score.get("status") == "structural_issue"
        )

        if has_structural_anomalies:
            fixed_stem, stem_fixes = auto_fix_deterministic(stem)
            fixed_options = {}
            opt_fixes = []
            for opt_k, opt_v in options.items():
                fixed_v, fxs = auto_fix_deterministic(opt_v)
                fixed_options[opt_k] = fixed_v
                opt_fixes.extend(fxs)

            # Sort options canonically (A, B, C, D)
            if fixed_options:
                fixed_options = dict(sorted(fixed_options.items()))

            all_fixes = list(set(stem_fixes + opt_fixes))

            if all_fixes:
                block["stem"] = fixed_stem
                block["options"] = fixed_options

                # Re-evaluate quality after deterministic auto-fix
                new_full_text = fixed_stem + "\n" + "\n".join(
                    f"({k}) {v}" for k, v in fixed_options.items()
                )
                new_score = score_question_text(new_full_text, has_malayalam_expected=has_malayalam, page_flags=all_page_flags)
                quality_scores[q_num] = {
                    "coverage_pct": new_score.coverage_pct,
                    "unmapped_glyph_count": new_score.unmapped_glyph_count,
                    "suspicious_gap_count": new_score.suspicious_gap_count,
                    "structural_anomalies": new_score.structural_anomalies,
                    "validation_flags": new_score.validation_flags,
                    "status": new_score.status,
                }

                if new_score.status == "ok":
                    entry = {
                        "question_number": q_num,
                        "failure_type": failure_type,
                        "action": "auto_fix_deterministic",
                        "result": "fixed_tier0",
                        "applied_fixes": all_fixes,
                    }
                    repair_log.append(entry)
                    print(f"  Q{q_num}: auto_fix_deterministic resolved {all_fixes} (Tier 0, $0)", flush=True)
                    continue  # Resolved! Skip higher tiers.
                else:
                    # Update failure diagnosis for remaining issues (e.g. unmapped glyphs)
                    failure_type = diagnose_failure_type(
                        new_full_text, [], {},
                        unmapped_codes=all_unmapped if new_score.status == "font_issue" else None,
                        has_text_layer=any(p.get("has_text_layer", True) for p in pages),
                        validation_flags=new_score.validation_flags if new_score.status == "word_level_corruption" else None,
                    )
                    full_text = new_full_text

        entry = {
            "question_number": q_num,
            "failure_type": failure_type,
            "action": None,
            "result": None,
        }

        match failure_type:
            case "reordering_bug" | "band_merge_bug":
                # Structural issue persisted despite Tier 0 rules — log for review
                entry["action"] = "log_structural_bug"
                entry["result"] = "flagged_for_review"
                print(f"  Q{q_num}: {failure_type} — flagged for review (Tier 0, $0)", flush=True)

            case "equation_crop_fail":
                # Tier 1.5 — $0, re-detection would happen on re-extraction
                entry["action"] = "flag_equation_crop"
                entry["result"] = "flagged_for_recrop"
                print(f"  Q{q_num}: equation_crop_fail — flagged (Tier 1.5, $0)", flush=True)

            case "unmapped_glyph" | "unknown_font":
                # Tier 1 — batch glyphs for this font hash, single Vision call later
                entry["action"] = "batch_glyph_repair"
                # Find which font hash this question's unmapped codes belong to
                for fh_data in page_font_hashes.values():
                    for fname, fhash in fh_data.items():
                        if fhash not in font_unmapped_batches:
                            font_unmapped_batches[fhash] = {
                                "codes": [],
                                "font_name": fname,
                                "question_nums": [],
                                "question_texts": [],
                            }
                        font_unmapped_batches[fhash]["question_nums"].append(q_num)
                        font_unmapped_batches[fhash]["question_texts"].append(full_text)
                        # Add codes from all pages for this font
                        for codes_list in page_unmapped.values():
                            for code_entry in codes_list:
                                if code_entry.get("font", "").split("+")[-1] in fname or fname in code_entry.get("font", ""):
                                    font_unmapped_batches[fhash]["codes"].append(code_entry)
                        break
                    break  # Only need first match per question
                entry["result"] = "batched"
                print(f"  Q{q_num}: {failure_type} — batched for glyph repair (Tier 1)", flush=True)

            case "scan_quality" | "word_level_corruption":
                # Tier 2 — last resort, circuit breaker protected
                try:
                    from ..storage.config_manager import get_setting
                    q_ocr_enabled = get_setting("question_ocr_enabled", True)
                except Exception:
                    q_ocr_enabled = True

                if not q_ocr_enabled:
                    entry["action"] = "ocr_disabled"
                    entry["result"] = "skipped"
                    print(f"  Q{q_num}: {failure_type} — skipped (Question OCR disabled in settings)", flush=True)
                elif vision_calls >= max_ocr:
                    entry["action"] = "circuit_breaker_halt"
                    entry["result"] = "halted"
                    print(f"  Q{q_num}: {failure_type} — HALTED by circuit breaker "
                          f"({vision_calls}/{max_ocr} Vision calls used)", flush=True)
                else:
                    entry["action"] = "targeted_crop_ocr"
                    entry["result"] = "flagged_for_ocr"
                    vision_calls += 1
                    print(f"  Q{q_num}: {failure_type} — flagged for targeted OCR "
                          f"(Tier 2, call {vision_calls}/{max_ocr})", flush=True)

            case _:
                entry["action"] = "unknown"
                entry["result"] = "logged"
                print(f"  Q{q_num}: {failure_type} — unknown, logged", flush=True)

        repair_log.append(entry)

    # Execute batched glyph repairs (Tier 1) — one Vision call per font hash
    for font_hash, batch in font_unmapped_batches.items():
        if not batch["codes"]:
            continue

        enable_repair = os.getenv("FONT_REPAIR_BATCH_GLYPHS", "true").lower() in ("true", "1", "yes")
        if not enable_repair:
            print(f"[Repair] Font repair disabled. Skipping glyph repair for {batch['font_name']}", flush=True)
            continue

        print(f"[Repair] Executing batched glyph repair for font '{batch['font_name']}' "
              f"({len(batch['codes'])} unmapped codes, "
              f"{len(batch['question_nums'])} affected questions)", flush=True)

        # We need the font buffer — extract from the PDF
        file_path = state.get("file_path", "")
        font_buffer = None
        if file_path and os.path.exists(file_path):
            try:
                import pymupdf
                doc = pymupdf.open(file_path)
                for page_idx in range(len(doc)):
                    for fi in doc[page_idx].get_fonts(full=True):
                        xref = fi[0]
                        fname = fi[3]
                        try:
                            _, _, _, buf = doc.extract_font(xref)
                            if buf:
                                import hashlib
                                if hashlib.md5(buf).hexdigest() == font_hash:
                                    font_buffer = buf
                                    break
                        except Exception:
                            pass
                    if font_buffer:
                        break
            except Exception as e:
                print(f"[Repair] Failed to extract font buffer: {e}", flush=True)

        if font_buffer:
            result = repair_unmapped_glyphs(
                font_hash, batch["font_name"], font_buffer, batch["codes"]
            )

            if result.get("status") == "success" and result.get("proposed_mapping"):
                # Validate before committing
                committed = validate_and_commit_font_map(
                    font_hash, batch["font_name"], font_buffer,
                    result["proposed_mapping"],
                    batch["question_texts"],
                )
                vision_calls += 1
                action_result = "committed" if committed else "rejected"
                print(f"[Repair] Font map for '{batch['font_name']}': {action_result}", flush=True)

                for qn in batch["question_nums"]:
                    for entry in repair_log:
                        if entry.get("question_number") == qn and entry.get("action") == "batch_glyph_repair":
                            entry["result"] = action_result
            else:
                print(f"[Repair] Glyph repair failed for '{batch['font_name']}': "
                      f"{result.get('message', 'unknown error')}", flush=True)

    # Update token usage
    total_p = token_usage.get("prompt_tokens", 0)
    total_c = token_usage.get("completion_tokens", 0)

    print(f"[Repair] Complete. {vision_calls} Vision calls used.", flush=True)

    return {
        "extracted_blocks": blocks,
        "quality_scores": quality_scores,
        "repair_log": repair_log,
        "vision_calls_count": vision_calls,
        "token_usage": {
            "prompt_tokens": total_p,
            "completion_tokens": total_c,
            "total_tokens": total_p + total_c,
        }
    }



def rule_check_node(state: GraphState) -> GraphState:
    """Filters out malformed questions and dedups."""
    blocks = state.get("extracted_blocks", [])
    valid_blocks = []
    seen_numbers = set()
    
    for b in blocks:
        if not isinstance(b, dict):
            continue
        q_num = b.get("question_number")
        if q_num in seen_numbers:
            continue
            
        options = b.get("options", {})
        marked_ans = b.get("marked_answer")
        
        if isinstance(options, dict) and options and marked_ans:
            if marked_ans not in options:
                found = False
                for k in options:
                    if str(k).upper() == str(marked_ans).upper():
                        b["marked_answer"] = k
                        found = True
                        break
                if not found:
                    b["marked_answer"] = marked_ans
                
        seen_numbers.add(q_num)
        valid_blocks.append(b)
        
    print(f"Rule check validated {len(valid_blocks)} out of {len(blocks)} blocks.", flush=True)
    return {"extracted_blocks": valid_blocks}

def rescan_question_vision(image_bytes: bytes, question_number: int, engine: Optional[str] = None):
    """Uses Groq Vision, Gemini Vision, or Local Tesseract OCR to re-extract a specific question accurately."""
    import base64
    import io

    # Resolve active OCR engine & enabled status
    try:
        from ..storage.config_manager import get_setting
        ocr_enabled = get_setting("question_ocr_enabled", True)
        if not ocr_enabled and engine is None:
            raise ValueError("Whole Question OCR is disabled in Pipeline Settings.")
        if engine is None:
            engine = get_setting("question_ocr_engine", "groq")
    except Exception as err:
        if "disabled in Pipeline Settings" in str(err):
            raise
    engine = str(engine or "groq").strip().lower()
    if engine in ("off", "disabled"):
        raise ValueError("Whole Question OCR is disabled.")

    # --- Engine 1: Tesseract Local ($0, offline Malayalam & English) ---
    if engine == "tesseract":
        try:
            import pytesseract
            from PIL import Image
            pil_img = Image.open(io.BytesIO(image_bytes))

            # Detect available Malayalam traineddata
            try:
                avail_langs = pytesseract.get_languages()
            except Exception:
                avail_langs = []

            tess_lang = "mal+eng" if "mal" in avail_langs else "eng"
            raw_text = pytesseract.image_to_string(pil_img, lang=tess_lang, config="--psm 6").strip()

            # Separate stem, options, marked answer, explanation
            stem, options, marked_ans, explanation = separate_stem_and_options(raw_text)

            # Strip leading question number if present in stem (e.g. '12.' or '12:-')
            stem = re.sub(rf'^(?:Question\s*|Q\.?\s*)?{question_number}\s*[\.\):-]\s*', '', stem, flags=re.IGNORECASE).strip()

            data = {
                "question_number": question_number,
                "stem": stem or raw_text,
                "options": options or {},
                "marked_answer": marked_ans,
                "explanation": explanation
            }
            return data, 0, 0
        except Exception as e:
            print(f"[Question OCR] Tesseract OCR failed: {e}", flush=True)
            raise

    prompt = """
Extract ONLY Question {question_number} from this page exactly as it appears.
CRITICAL INSTRUCTIONS:
1. Transcribe and preserve all mathematical equations, fractions, formulas, and symbols into standard inline LaTeX ($...) or display LaTeX ($$...$$), just like non-image math equations (e.g. $\frac{1}{2}$, $\sqrt{x^2+y^2}$, $x^2$, $\chi^2$). For multiplication, ALWAYS use LaTeX `\times` (e.g., $96 = 2^5 \times 3$, $404 = 2^2 \times 101$). Never use the letter 'x' or 'x^3' for multiplication! Preserve punctuation (commas, periods) between separate equations. Never output raw image tags or placeholders for mathematical formulas.
2. Extract all options into 'options': {"A": "...", "B": "...", "C": "...", "D": "..."}.
   - NEVER include options (e.g. A:-, B:-, C:-, D:-) or option contents inside the question 'stem'.
3. Extract the given 'Correct Answer' into 'marked_answer' if present (e.g. if an option letter or option text is highlighted in yellow/color, marked with a tick mark, circled, or explicitly printed as answer / ഉത്തരം). Set to null if not marked.
4. Extract the given 'Solution', 'Explanation', or 'Description' into 'explanation' if present below the question or options (e.g. 'Solution:', 'Explanation:', 'Description:', 'വിശദീകരണം:', 'പരിഹാരം:'). Set to null if not present in the document.
5. For matching questions (List-I / List-II), preserve the two-column table using tabs or clear alignment. Keep matching option codes (e.g. '(a) (b) (c) (d) (e)') together on one line within each option, DO NOT break them into individual single-letter lines.
6. Transcribe Malayalam and Indic text faithfully into standard Unicode script. Do not invent or summarize.

Output ONLY a valid JSON object exactly matching this schema, with no markdown code blocks or extra text:
{
  "question_number": {question_number},
  "stem": "Question stem text here\\na. $\\chi^2$ test\\nb. 't' test",
  "options": {
    "A": "Option A text",
    "B": "Option B text",
    "C": "Option C text",
    "D": "Option D text"
  },
  "marked_answer": "B",
  "explanation": "Solution or explanation text here"
}
""".replace("{question_number}", str(question_number))
    # --- Engine 2: Groq Vision ---
    if engine == "groq" or (engine not in ("gemini", "tesseract") and is_groq_enabled()):
        if is_groq_enabled():
            try:
                from groq import Groq
                groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
                client = Groq(api_key=groq_key)
                b64_img = base64.b64encode(image_bytes).decode("utf-8")
                model_name = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
                chat_completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64_img}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=600,
                    response_format={"type": "json_object"}
                )
                resp_text = chat_completion.choices[0].message.content
                cleaned = _clean_json_output(resp_text)
                data = json.loads(cleaned)
                p_tok = chat_completion.usage.prompt_tokens if chat_completion.usage else 0
                c_tok = chat_completion.usage.completion_tokens if chat_completion.usage else 0
                return data, p_tok, c_tok
            except Exception as e:
                print(f"Groq vision rescan failed: {e}", flush=True)
                if engine == "groq":
                    raise

    # --- Engine 3: Gemini Vision ---
    if engine == "gemini" or is_gemini_enabled():
        if is_gemini_enabled():
            google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
            client = genai.Client(api_key=google_key)
            model_name = os.getenv("GEMINI_EXTRACT_MODEL", "gemini-3.1-flash") 
            res = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt
                ],
                config=types.GenerateContentConfig(temperature=0.1)
            )
            cleaned = _clean_json_output(res.text)
            data = json.loads(cleaned)
            p_tok = res.usage_metadata.prompt_token_count if res.usage_metadata else 0
            c_tok = res.usage_metadata.candidates_token_count if res.usage_metadata else 0
            return data, p_tok, c_tok

    groq_st = get_groq_status()
    gemini_st = get_gemini_status()
    raise ValueError(f"Selected question OCR engine '{engine}' is unavailable. Groq: {groq_st['reason']}; Gemini: {gemini_st['reason']}")

from concurrent.futures import ThreadPoolExecutor, as_completed


def _solve_single_block(b: dict, total: int = 1) -> dict:
    """Validates/solves a single question block using configured LLM provider."""
    q_num = b.get("question_number", "?")
    stem = b.get("stem", "")
    options = b.get("options", {})
    marked = b.get("marked_correct_answer") or b.get("marked_answer")

    try:
        from ..storage.config_manager import get_setting
        validation_enabled = get_setting("validation_enabled", True)
        provider = get_setting("validation_provider", "groq")
    except Exception:
        validation_enabled = True
        provider = "groq"

    if not validation_enabled or not is_llm_enabled():
        return {
            **b,
            "marked_correct_answer": marked,
            "marked_answer": marked,
            "validated_answer": None,
            "is_valid": False,
            "confidence": "none",
            "reasoning": "Answer validation disabled or AI provider unavailable.",
            "used_web_search": False,
            "_p_tokens": 0,
            "_c_tokens": 0
        }

    solve_llm = get_solve_llm(provider)
    if not solve_llm:
        return {
            **b,
            "marked_correct_answer": marked,
            "marked_answer": marked,
            "validated_answer": None,
            "is_valid": False,
            "confidence": "none",
            "reasoning": f"No LLM client available for provider '{provider}'.",
            "used_web_search": False,
            "_p_tokens": 0,
            "_c_tokens": 0
        }

    prompt = SOLVE_PROMPT.replace("{stem}", stem).replace("{options}", json.dumps(options))
    p_tokens, c_tokens = 0, 0
    try:
        from langchain_core.messages import HumanMessage
        resp = solve_llm.invoke([HumanMessage(content=prompt)])
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            p_tokens += resp.usage_metadata.get("input_tokens", 0)
            c_tokens += resp.usage_metadata.get("output_tokens", 0)
            
        cleaned = _clean_json_output(resp.content)
        solve_result = json.loads(cleaned)
        
        validated_answer = solve_result.get("validated_answer")
        confidence = solve_result.get("confidence", "medium")
        reasoning = solve_result.get("reasoning", "")
        
        is_valid = (str(validated_answer).strip().upper() == str(marked).strip().upper()) if marked and validated_answer else False
        return {
            **b,
            "marked_correct_answer": marked,
            "marked_answer": marked,
            "validated_answer": validated_answer,
            "is_valid": is_valid,
            "confidence": confidence,
            "reasoning": reasoning,
            "used_web_search": False,
            "_p_tokens": p_tokens,
            "_c_tokens": c_tokens
        }
    except Exception as e:
        print(f"Solve failed for Q{q_num}: {e}", flush=True)
        return {
            **b,
            "validated_answer": None,
            "is_valid": False,
            "confidence": "low",
            "reasoning": f"Error validating answer: {e}",
            "used_web_search": False,
            "_p_tokens": p_tokens,
            "_c_tokens": c_tokens
        }

