import json
import os
import re
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
from .state import GraphState, QuestionBlock, ProcessedQuestion
from ..prompts.templates import EXTRACT_PROMPT, SOLVE_PROMPT

def get_gemini_status() -> dict:
    """Returns detailed status of Gemini integration."""
    enable = os.getenv("ENABLE_GEMINI", "true").lower() in ("true", "1", "yes")
    key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not enable:
        return {"enabled": False, "reason": "Disabled in .env (ENABLE_GEMINI=false)"}
    if not key:
        return {"enabled": False, "reason": "GOOGLE_API_KEY is not set"}
    return {"enabled": True, "reason": "Active"}


def is_gemini_enabled() -> bool:
    """Checks if Gemini API integration is explicitly enabled and API key is available."""
    return get_gemini_status()["enabled"]


def get_extract_llm():
    if not is_gemini_enabled():
        return None
    try:
        model_name = os.getenv("GEMINI_EXTRACT_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
        key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=key
        )
    except Exception as e:
        print(f"Warning: Failed to initialize extract_llm: {e}", flush=True)
        return None


def get_solve_llm():
    if not is_gemini_enabled():
        return None
    try:
        model_name = os.getenv("GEMINI_SOLVE_MODEL", "gemini-3.1-pro-preview")
        key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=key
        )
    except Exception as e:
        print(f"Warning: Failed to initialize solve_llm: {e}", flush=True)
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

def extract_node(state: GraphState) -> GraphState:
    """Extracts question blocks from raw text using a lightweight Regex parser."""
    raw_text = state.get("raw_text", "")
    extracted = []
    
    # 1. Determine if the document explicitly uses 'Question' or 'Q' markers
    explicit_matches = list(re.finditer(r'(?i)(?:^|\n)\s*(?:Question\s*|Q\.?\s*)(\d+)\s*(?::\-?|[\.\):-])\s*', raw_text))
    if len(explicit_matches) >= 3:
        # Document uses explicit Question/Q markers (e.g. 'Question 1:-', 'Question20:-', 'Q.1:')
        # Strictly require Question/Q prefix so numbers inside options/stems (like '52-ാം', dates, etc.) are never treated as question delimiters!
        q_prefix_pattern = r'(?:Question\s*|Q\.?\s*)\d+\s*(?::\-?|[\.\):-])\s*'
        q_extract_pattern = r'^(?:Question\s*|Q\.?\s*)(\d+)\s*(?::\-?|[\.\):-])\s*(.*)'
    else:
        # Fallback for documents with simple numeric numbering (e.g. '1. ', '2. ', '1) ')
        # Require delimiter to be followed by whitespace; NEVER treat solitary hyphen '-' without Question as a question delimiter!
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
        
        # Extract options securely by finding the true sequential options block (A, B, C, D)
        # Strictly require uppercase [A-D] preceded by newline/whitespace/tab to avoid false positives
        # from internal sub-items like (a), (b), (c), (d), (e) or statements in matching questions!
        opt_pattern = re.compile(r'(?:^|\n|\s{2,}|\t)(?:\()?([A-D])(?:\)|\.|:-)\s*')
        matches = list(opt_pattern.finditer(content))
        
        best_sequence = []
        for i, m in enumerate(matches):
            if m.group(1) == "A":
                curr_seq = [m]
                expected_next = "B"
                for next_m in matches[i+1:]:
                    if next_m.group(1) == expected_next:
                        curr_seq.append(next_m)
                        if expected_next == "B":
                            expected_next = "C"
                        elif expected_next == "C":
                            expected_next = "D"
                        elif expected_next == "D":
                            break
                if len(curr_seq) > len(best_sequence) or (len(curr_seq) >= 2 and len(curr_seq) >= len(best_sequence)):
                    best_sequence = curr_seq

        stem = content.strip()
        options = {}
        marked_answer = None

        if len(best_sequence) >= 2:
            stem = content[:best_sequence[0].start()].strip()
            for idx, m in enumerate(best_sequence):
                letter = m.group(1)
                start_pos = m.end()
                end_pos = best_sequence[idx+1].start() if idx + 1 < len(best_sequence) else len(content)
                opt_text = content[start_pos:end_pos].strip()
                
                ans_match = re.search(r'(?:Ans(?:wer)?|Correct(?:\s*Answer)?)\s*[:\-]*\s*(?:is\s*)?(?:Option\s*[:\-]*\s*)?([A-D])\b', opt_text, re.IGNORECASE)
                if ans_match:
                    marked_answer = ans_match.group(1).upper()
                    opt_text = opt_text[:ans_match.start()].strip()
                options[letter] = opt_text
        else:
            # Fallback for questions where options might not strictly start with newline
            fallback_pattern = re.compile(r'(?:^|\s)(?:\()?([A-D])(?:\)|\.|:-)\s*')
            fb_matches = list(fallback_pattern.finditer(content))
            if fb_matches:
                last_positions = {}
                for m in fb_matches:
                    last_positions[m.group(1)] = m
                if 'A' in last_positions and 'B' in last_positions:
                    true_option_matches = sorted(last_positions.values(), key=lambda m: m.start())
                    stem = content[:true_option_matches[0].start()].strip()
                    for i, match in enumerate(true_option_matches):
                        opt_letter = match.group(1)
                        start_idx = match.end()
                        end_idx = true_option_matches[i+1].start() if i + 1 < len(true_option_matches) else len(content)
                        opt_text = content[start_idx:end_idx].strip()
                        ans_match = re.search(r'(?:Ans(?:wer)?|Correct(?:\s*Answer)?)\s*[:\-]*\s*(?:is\s*)?(?:Option\s*[:\-]*\s*)?([A-D])\b', opt_text, re.IGNORECASE)
                        if ans_match:
                            marked_answer = ans_match.group(1).upper()
                            opt_text = opt_text[:ans_match.start()].strip()
                        options[opt_letter] = opt_text
        
        # Check if marked answer was in the stem
        if not marked_answer:
            ans_match = re.search(r'(?:Ans(?:wer)?|Correct(?:\s*Answer)?)\s*[:\-]*\s*(?:is\s*)?(?:Option\s*[:\-]*\s*)?([A-D])\b', stem, re.IGNORECASE)
            if ans_match:
                marked_answer = ans_match.group(1).upper()
                stem = stem[:ans_match.start()].strip()
        
        extracted.append({
            "question_number": q_num,
            "stem": stem,
            "options": options,
            "marked_answer": marked_answer
        })
        
    print(f"Regex parser extracted {len(extracted)} questions.", flush=True)
    
    avg_length = sum(len(b["stem"]) for b in extracted) / max(1, len(extracted))
    
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
                
                if "token_usage" not in state:
                    state["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                state["token_usage"]["prompt_tokens"] += p_tokens
                state["token_usage"]["completion_tokens"] += c_tokens
                state["token_usage"]["total_tokens"] += (p_tokens + c_tokens)
                
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

        if not extracted:
            extracted = []

    return {"extracted_blocks": extracted, "token_usage": state.get("token_usage", {})}

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
        
        if not isinstance(options, dict) or not options:
            continue
            
        if marked_ans and marked_ans not in options:
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

def rescan_question_vision(image_bytes: bytes, question_number: int):
    """Uses Gemini Vision to re-extract a specific question accurately."""
    status = get_gemini_status()
    if not status["enabled"]:
        raise ValueError(f"Gemini is disabled: {status['reason']}")
        
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    client = genai.Client(api_key=google_key)
    
    prompt = f"""
Extract ONLY Question {question_number} from this page exactly as it appears.
CRITICAL INSTRUCTIONS:
1. Preserve all mathematical symbols, formulas, and Greek letters as standard inline LaTeX ($...$).
2. Extract all options into 'options': {{"A": "...", "B": "...", "C": "...", "D": "..."}}.
   - NEVER include options (e.g. A:-, B:-, C:-, D:-) or option contents inside the question 'stem'.
3. Extract the given 'Correct Answer' into 'marked_answer' if present.
4. For matching questions (List-I / List-II), preserve the two-column table using tabs or clear alignment. Keep matching option codes (e.g. '(a) (b) (c) (d) (e)') together on one line within each option, DO NOT break them into individual single-letter lines.

Output ONLY a valid JSON object exactly matching this schema, with no markdown code blocks or extra text:
{{
  "question_number": {question_number},
  "stem": "Question stem text here\\na. $\\\\chi^2$ test\\nb. 't' test",
  "options": {{
    "A": "Option A text",
    "B": "Option B text",
    "C": "Option C text",
    "D": "Option D text"
  }},
  "marked_answer": "B"
}}
"""
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

from concurrent.futures import ThreadPoolExecutor, as_completed

def _solve_single_block(b: dict, total: int) -> dict:
    q_num = b.get("question_number", "?")
    stem = b.get("stem", "")
    options = b.get("options", {})
    marked = b.get("marked_correct_answer") or b.get("marked_answer")

    status = get_gemini_status()
    solve_llm = get_solve_llm() if status["enabled"] else None
    if not solve_llm:
        print(f"Skipping AI validation for Q{q_num}: {status['reason']}", flush=True)
        return {
            **b,
            "marked_correct_answer": marked,
            "marked_answer": marked,
            "validated_answer": None,
            "is_valid": False,
            "confidence": "none",
            "reasoning": f"Gemini AI is disabled ({status['reason']}). AI-powered answer validation was not performed.",
            "used_web_search": False,
            "_p_tokens": 0,
            "_c_tokens": 0
        }

    print(f"Solving Q{q_num} (out of {total}) with Gemini...", flush=True)
    prompt = SOLVE_PROMPT.replace("{stem}", stem).replace("{options}", json.dumps(options))
    p_tokens, c_tokens = 0, 0

    try:
        resp = solve_llm.invoke([HumanMessage(content=prompt)])
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            p_tokens += resp.usage_metadata.get("input_tokens", 0)
            c_tokens += resp.usage_metadata.get("output_tokens", 0)
            
        cleaned = _clean_json_output(resp.content)
        solve_result = json.loads(cleaned)
        
        validated_answer = solve_result.get("validated_answer")
        confidence = solve_result.get("confidence", "low")
        reasoning = solve_result.get("reasoning", "")
        used_web_search = False
        
        # Web search fallback if confidence is low
        if str(confidence).lower() == "low":
            used_web_search = True
            try:
                with DDGS() as ddgs:
                    results = [r for r in ddgs.text(stem, max_results=3)]
                    search_context = "\n".join([r.get("body", "") for r in results])
                    
                if search_context.strip():
                    retry_prompt = f"Context from web:\n{search_context}\n\n{prompt}"
                    retry_resp = solve_llm.invoke([HumanMessage(content=retry_prompt)])
                    if hasattr(retry_resp, "usage_metadata") and retry_resp.usage_metadata:
                        p_tokens += retry_resp.usage_metadata.get("input_tokens", 0)
                        c_tokens += retry_resp.usage_metadata.get("output_tokens", 0)
                        
                    retry_cleaned = _clean_json_output(retry_resp.content)
                    retry_result = json.loads(retry_cleaned)
                    
                    validated_answer = retry_result.get("validated_answer", validated_answer)
                    confidence = retry_result.get("confidence", confidence)
                    reasoning = f"[Web Search Used] {retry_result.get('reasoning', reasoning)}"
            except Exception as search_err:
                print(f"Web search fallback failed for Q{q_num}: {search_err}", flush=True)
        
        marked = b.get("marked_correct_answer") or b.get("marked_answer")
        is_valid = (str(validated_answer).strip().upper() == str(marked).strip().upper()) if marked and validated_answer else False
        status_str = "MATCH" if is_valid else f"FLAGGED (marked={marked}, validated={validated_answer})"
        print(f"Done Q{q_num}: {status_str}", flush=True)
        
        return {
            **b,
            "marked_correct_answer": marked,
            "marked_answer": marked,
            "validated_answer": validated_answer,
            "is_valid": is_valid,
            "confidence": confidence,
            "reasoning": reasoning,
            "used_web_search": used_web_search,
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
            "reasoning": f"Error solving question: {str(e)}",
            "used_web_search": False,
            "_p_tokens": p_tokens,
            "_c_tokens": c_tokens
        }

def solve_and_validate_node(state: GraphState) -> GraphState:
    """Solves blocks concurrently using stronger model and compares with marked answer."""
    blocks = state.get("extracted_blocks", [])
    total = len(blocks)
    print(f"Starting parallel solve for {total} questions...", flush=True)
    
    processed = []
    total_p = state.get("token_usage", {}).get("prompt_tokens", 0)
    total_c = state.get("token_usage", {}).get("completion_tokens", 0)
    
    # Use 5 concurrent workers for quick solving
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_solve_single_block, b, total): b for b in blocks}
        for future in as_completed(futures):
            res = future.result()
            total_p += res.pop("_p_tokens", 0)
            total_c += res.pop("_c_tokens", 0)
            processed.append(res)
            
    # Sort by question number to preserve document order
    processed.sort(key=lambda x: x.get("question_number", 0))
    
    gemini_on = is_gemini_enabled()
    summary = {
        "total": len(processed),
        "validated": sum(1 for p in processed if p.get("is_valid")),
        "flagged": sum(1 for p in processed if (p.get("validated_answer") and not p.get("is_valid"))),
        "unvalidated": sum(1 for p in processed if not p.get("validated_answer")),
        "gemini_enabled": gemini_on
    }
    
    print(f"Solve complete: {summary['validated']} validated, {summary['flagged']} flagged.", flush=True)
    return {
        "processed_questions": processed, 
        "summary": summary,
        "token_usage": {
            "prompt_tokens": total_p,
            "completion_tokens": total_c,
            "total_tokens": total_p + total_c
        }
    }
