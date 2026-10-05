import json
import os
import re
import traceback
from typing import Optional, Dict, List, Tuple, Any
import time
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
Extract all questions from this uploaded PDF document into a structured JSON array for STUDENTS studying for their exams.
The extracted questions, options, and explanations MUST make complete, clear pedagogical and grammatical sense to a student.

CRITICAL STUDENT-CENTRIC READABILITY & MATH FORMATTING RULES:
- THE USERS ARE STUDENTS: All text must read smoothly, clearly, and naturally.
- DO NOT PUT REGULAR TEXT, SENTENCES, OR SUBQUESTION LABELS INSIDE LATEX MATH DELIMITERS ($...$ or $$...$$):
  * LaTeX math mode strips all spaces and renders words in unreadable slanted italics.
  * Explanatory prose, subquestion labels like "(a)" or "(b)", and connecting words MUST remain normal text OUTSIDE math delimiters.
  * WRONG: `$\text{(b) The reactance is given by } X_C = \frac{1}{\omega C}$`
  * CORRECT: `(b) The reactance is given by $X_C = \frac{1}{\omega C}$`
  * EXCEPTION for Equations: If an equation inherently contains words or subscripts as words (e.g. "Power = ", "Work", or $i_{\mathrm{rms}}$), you MUST use `\text{...}` or `\mathrm{...}` inside the math block exactly as they appear in the PDF so they render in the correct font (e.g., $i_{\mathrm{rms}} = \frac{V_{\mathrm{rms}}}{R}$ or $\text{Power} = 400\text{ W}$). Do not break the equation just to keep the word out.
- DO NOT LABEL ANYTHING IN ANSWERS AS SUBQUESTION: Numbered steps or items in answers/solutions are ANSWER/SOLUTION STEPS, NOT sub-questions! Keep all solution steps in 'explanation' / 'answer_text'. NEVER put answer steps into 'sub_parts'.

CRITICAL INSTRUCTIONS:
1. Preserve all mathematical symbols, formulas, and Greek letters as standard inline LaTeX ($...$).
   - For example: chi-squared must be written as $\\chi^2$, t-test as 't' test or $t$-test.
2. DIAGRAM & IMAGE POSITIONING:
   - Diagrams may appear before any sub-question starts (introductory question stem), inside a specific sub-question, as an option choice in an MCQ, or in the explanation/solution.
   - Preserve all image tokens (e.g., {{IMG:https://...}} or {{IMG_BLOCK:https://...}}) EXACTLY at their original position. NEVER move or delete them.
3. Determine the structure of the question:
   - "mcq": Has explicit multiple-choice options (e.g., A/B/C/D or 1/2/3/4). If options/choices are listed, it MUST be "mcq".
   - "descriptive_single": A standard descriptive question with no sub-parts.
   - "descriptive_subquestions": A descriptive question containing multiple sub-parts (e.g., (a), (b), (c) or (i), (ii)).
4. For "mcq" questions:
   - Extract options into 'options': {"A": "...", "B": "...", "C": "...", "D": "..."}. If an option contains an image token, keep the {{IMG:...}} token inside that option.
   - CRITICAL (DIFFERENTIATING OPTION LABELS FROM WORDS): When an option starts with a word that begins with the same letter as the option (e.g. Option B is "(b) Both A and B", Option A is "(a) All of the above"): NEVER strip the initial letter from the word! Option B's text MUST be "Both A and B" (NEVER "oth A and B"). Option A's text MUST be "All of the above" (NEVER "ll of the above"). Differentiate clearly between the option label (B) and the text (Both).
   - Extract the correct answer into 'marked_answer' (e.g., "A", "B"). DO NOT solve the question yourself to guess the answer. ONLY extract it if it is explicitly provided (e.g., in an Answer Key or if the explanation starts with "(C)"). Set to null if not explicitly marked.
   - 'sub_parts': null
5. For "descriptive_subquestions":
   - Extract subparts into 'sub_parts': {"a": "...", "b": "..."}.
   - If there is introductory text/diagram before (a), keep it in 'stem'. If the question starts directly with (a), set 'stem' to "".
   - Leave 'options': {}. Set 'marked_answer': null.
   - Extract solution into 'answer_text' / 'explanation'.
6. For "descriptive_single":
   - Leave 'options': {}. Set 'sub_parts': null, 'marked_answer': null.
7. STRICT NEGATIVE CONSTRAINTS:
   - ONLY extract actual examination questions. Question numbers must be positive integers.
   - NEVER extract document titles, exam headers, or syllabus outlines as questions!
   - NEVER extract or include marks indicators like "[1 mark]" or "(2 marks)" in the extracted output.
8. PHYSICS / SCIENCE EQUATION RULES:
   - Vector notation: Use $\\hat{i}$, $\\hat{j}$, $\\hat{k}$ for unit vectors (î, ĵ, k̂).
   - Greek letters MUST be in LaTeX: ε₀ → $\\epsilon_0$, σ → $\\sigma$, τ → $\\tau$, φ → $\\phi$, θ → $\\theta$, μ → $\\mu$.
   - Scientific notation: 10⁻⁷ → $10^{-7}$, 26.4 × 10⁻¹² → $26.4 \\times 10^{-12}$.
   - Fractions in options: a/b → $\\frac{a}{b}$, √2kqq₀/a² → $\\frac{\\sqrt{2}kqq_0}{a^2}$.
   - Units with negative exponents: NC⁻¹ → $NC^{-1}$, Nm²C⁻¹ → $Nm^2C^{-1}$.
   - Subscripts: q₀, r₁, r₂, ε₀ → $q_0$, $r_1$, $r_2$, $\\epsilon_0$.
   - Superscripts: m², cm², C/m² → $m^2$, $cm^2$, $C/m^2$.
9. ASSERTION-REASON QUESTION RULES:
   - If a question has "Assertion (A):" and "Reason (R):" followed by (a)-(d) options about whether Assertion/Reason are true, it is ALWAYS "mcq".
   - Put the full Assertion and Reason statements in "stem".
   - Options (a)-(d) contain the truth-value evaluation choices (e.g. "If both Assertion and Reason are true...").
   - NEVER classify Assertion-Reason questions as "descriptive_subquestions".

Output ONLY a valid JSON array matching this schema:
[
  {
    "question_number": 1,
    "question_type": "mcq",
    "stem": "Question stem text here\\na. $\\\\chi^2$ test\\nb. 't' test",
    "options": {
      "A": "Option A text",
      "B": "Option B text"
    },
    "sub_parts": null,
    "marked_answer": "B",
    "answer_text": null
  },
  {
    "question_number": 2,
    "question_type": "descriptive_subquestions",
    "stem": "Two point charges are placed...\\n{{IMG_BLOCK:https://...}}",
    "options": {},
    "sub_parts": {
      "a": "Draw the surfaces.",
      "b": "Why do they get close?"
    },
    "marked_answer": null,
    "answer_text": "(a) Equipotential surfaces:\\n{{IMG:https://...}}\\n(b) Because the field is stronger."
  },
  {
    "question_number": 3,
    "question_type": "descriptive_single",
    "stem": "Calculate the length of each of the side.",
    "options": {},
    "sub_parts": null,
    "marked_answer": null,
    "answer_text": "Let l = length of each side..."
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
    from ..ingest.option_answer_separator import parse_trailing_meta as _parse_trailing_meta
    return _parse_trailing_meta(text)


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

    # Formatting wrappers must not hide structural option markers from parsing.
    content = re.sub(r'<(?:strong|b|em|i|mark)\b[^>]*>([\s\S]*?)</(?:strong|b|em|i|mark)>', r'\1', content, flags=re.IGNORECASE)
    content = re.sub(r'(?<!\w)(?:\*{1,3}|_{1,3})([^\n]*?)(?:\*{1,3}|_{1,3})(?!\w)', r'\1', content)

    # Heal any accidental math conversions of option markers (e.g. 1\nB:- turned into \frac{1}{B}:-)
    content = re.sub(
        r'(?:\$?\{\\frac\{1\}\{([A-Ha-h])\}\}?|\$?\\\s*frac\s*\{\s*1\s*\}\s*\{\s*([A-Ha-h])\s*\}\$?|1\s*/\s*([A-Ha-h]))(\s*[:-]+|\s*\.|\s*\))',
        lambda m: f"1\n{(m.group(1) or m.group(2) or m.group(3)).upper()}{m.group(4)}",
        content
    )

    # Option detection pattern:
    # 1. Parenthesized markers: (A), (a), (1) - can be preceded by start of line, newline, or whitespace
    # 2. Bare letter markers: A., B., A:- - must be at line start, newline, or after 2+ spaces / sentence punctuation
    opt_pattern = re.compile(
        r'(?:(?:^|\n|\s+)\(([A-Ea-e1-4])\)[\.\:\-]?\s*)|'
        r'(?:(?:^|\n|[ \t]+|\.\s+)([A-Ea-e1-4])[\.\)\:-]\s+)'
    )
    matches = []
    for m in opt_pattern.finditer(content):
        letter = m.group(1) or m.group(2)
        start_idx = m.start(1) if m.group(1) else m.start(2)
        prefix = content[max(0, start_idx - 35):start_idx].lower()
        full_match = m.group(0).strip()
        has_paren = full_match.startswith("(")

        # 0. Reject if followed by equation operator or equal sign (e.g. A = 240, A = 120, A + B)
        # CRITICAL: ONLY reject bare letters without parentheses (e.g. 'A = 240', 'A + B').
        # Parenthesized markers like '(B)' or '(D)' can validly have values starting with '+5 V', '-10 J', '\sqrt{c}', etc.!
        post_match = content[m.end():m.end() + 10].strip()
        if not has_paren and re.match(r'^[\=\+\-\*\/\^\_\±\×\÷\√]', post_match):
            continue
        elif has_paren and post_match.startswith('='):
            continue

        # 1. Reject candidate matches preceded by Assertion/Reason/Statement labels:
        if re.search(r'(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)[\s\:\-\*\_\(]*$', prefix, re.IGNORECASE):
            continue
        # 2. Reject candidate matches preceded by prepositions / linkers / points / math variable contexts:
        #    ONLY reject bare letters (e.g. 'between points A and B', '0.2 A.', 'each of A = 120').
        #    Parenthesized markers like '(B)' should NEVER be rejected by words like 'to', 'of', 'or'.
        prev_word_match = re.search(r'([a-zA-Z]+)\s*(?:and\s*)?$', prefix)
        if prev_word_match:
            prev_word = prev_word_match.group(1).lower()
            if not has_paren:
                reject_words = {'both', 'between', 'from', 'and', 'or', 'is', 'are', 'was', 'were',
                               'if', 'whether', 'that', 'when', 'where', 'of', 'in', 'for', 'with',
                               'junction', 'terminal', 'point', 'points', 'cell', 'cells',
                               'charge', 'charges', 'plate', 'plates', 'to', 'number', 'mass',
                               'each', 'fragment', 'fragments', 'side', 'radius', 'amplitude'}
                if len(prev_word) >= 2 and prev_word in reject_words:
                    continue
            else:
                structural_reject = {'point', 'points', 'junction', 'terminal', 'cell', 'cells', 'number', 'mass'}
                if prev_word in structural_reject:
                    continue
        # 3. If bare letter (no parens), ensure it is not preceded by a number (scientific unit like '0.2 A.' or '5 V.')
        if not has_paren and re.search(r'\d+(?:\.\d+)?\s*$', prefix):
            continue
        # 3.5. Reject acronyms like A.C., D.C., A.M., P.M.
        if not has_paren:
            match_str_no_space = m.group(0).rstrip()
            post_str = content[m.end():m.end() + 10]
            if match_str_no_space.endswith('.') and re.match(r'^[a-zA-Z]', content[m.end():m.end()+1]):
                continue
            if match_str_no_space.endswith('.') and letter.upper() in ('A', 'D', 'P', 'B', 'N'):
                if re.match(r'^\s*[CcMmDd]\.', post_str, re.IGNORECASE):
                    continue
                if re.match(r'^\s*[Cc]\s', post_str, re.IGNORECASE) and letter.upper() in ('A', 'D'):
                    continue
        # 4. If match is immediately followed by Assertion / Reason label (e.g. "(A) Assertion:" or "(A): Assertion"):
        post_snippet = content[m.end():m.end() + 25].strip().lower()
        if re.match(r'^(?:[\:\.\-]+\s*)?(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)\b', post_snippet, re.IGNORECASE):
            continue
        matches.append((letter, m))
    
    best_sequence = []
    for i, m_info in enumerate(matches):
        letter = m_info[0].upper()
        if letter != 'A' and letter != '1':
            continue
        curr_seq = [m_info]
        curr_letter = letter
        for next_m in matches[i+1:]:
            next_letter = next_m[0].upper()
            
            if next_letter > curr_letter:
                distance = next_m[1].start() - curr_seq[-1][1].end()
                
                # If they are absurdly far apart (e.g. > 1500 chars), it's probably a false positive list
                if distance > 1500:
                    continue
                    
                # To prevent matching random A and then a random D later, ensure gap is small
                letter_gap = ord(next_letter) - ord(curr_letter)
                if letter_gap <= 2: # Allow skipping at most 1 letter (e.g. A -> C if B is mangled)
                    curr_seq.append(next_m)
                    curr_letter = next_letter
        
        if len(curr_seq) >= 2 and len(curr_seq) > len(best_sequence):
            best_sequence = curr_seq

    stem = content.strip()
    options = {}
    marked_answer = None
    explanation = None

    if len(best_sequence) >= 2:
        stem = content[:best_sequence[0][1].start()].strip()
        for idx, m_info in enumerate(best_sequence):
            # Retain original case of the match for better downstream classification
            letter = m_info[0]
            m = m_info[1]
            start_pos = m.end()
            end_pos = best_sequence[idx+1][1].start() if idx < len(best_sequence) - 1 else len(content)
            opt_text = content[start_pos:end_pos].strip()
            
            clean_opt, ans, exp = parse_trailing_meta(opt_text)
            if ans and not marked_answer:
                marked_answer = ans
            if exp and not explanation:
                explanation = exp
            options[letter] = clean_opt

    # Fallback 1: Flexible separator matching (must also start with A and follow ascending sequence)
    if len(options) < 2:
        fb_matches = []
        for m in opt_pattern.finditer(content):
            letter = m.group(1) or m.group(2)
            start_idx = m.start(1) if m.group(1) else m.start(2)
            prefix = content[max(0, start_idx - 35):start_idx].lower()
            full_match = m.group(0).strip()
            has_paren = full_match.startswith("(")
            # 1. Reject candidate matches preceded by Assertion/Reason/Statement labels:
            if re.search(r'(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)[\s\:\-\*\_\(]*$', prefix, re.IGNORECASE):
                continue
            # 2. Reject candidate matches preceded by prepositions / linkers / points:
            fb_prev_match = re.search(r'([a-zA-Z]+)\s*(?:and\s*)?$', prefix)
            if fb_prev_match:
                fb_prev_word = fb_prev_match.group(1).lower()
                if not has_paren:
                    fb_reject_words = {'both', 'between', 'from', 'and', 'or', 'is', 'are', 'was', 'were',
                                       'if', 'whether', 'that', 'when', 'where', 'of', 'in', 'for', 'with',
                                       'junction', 'terminal', 'point', 'points', 'cell', 'cells',
                                       'charge', 'charges', 'plate', 'plates', 'to'}
                    if len(fb_prev_word) >= 2 and fb_prev_word in fb_reject_words:
                        continue
                else:
                    structural_reject = {'point', 'points', 'junction', 'terminal', 'cell', 'cells'}
                    if fb_prev_word in structural_reject:
                        continue
            # 3. If bare letter (no parens), ensure it is not preceded by a number (scientific unit like '0.2 A.' or '5 V.')
            if not has_paren and re.search(r'\d+(?:\.\d+)?\s*$', prefix):
                continue
            # 3.5. Reject acronyms like A.C., D.C., A.M., P.M.
            if not has_paren:
                match_str_no_space = m.group(0).rstrip()
                post_str = content[m.end():m.end() + 10]
                if match_str_no_space.endswith('.') and re.match(r'^[a-zA-Z]', content[m.end():m.end()+1]):
                    continue
                if match_str_no_space.endswith('.') and letter.upper() in ('A', 'D', 'P', 'B', 'N'):
                    if re.match(r'^\s*[CcMmDd]\.', post_str, re.IGNORECASE):
                        continue
                    if re.match(r'^\s*[Cc]\s', post_str, re.IGNORECASE) and letter.upper() in ('A', 'D'):
                        continue
            # 4. If match is immediately followed by Assertion / Reason label (e.g. "(A) Assertion:" or "(A): Assertion"):
            post_snippet = content[m.end():m.end() + 25].strip().lower()
            if re.match(r'^(?:[\:\.\-]+\s*)?(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)\b', post_snippet, re.IGNORECASE):
                continue
            fb_matches.append((letter, m))
        if fb_matches:
            fb_best = []
            for i, m_info in enumerate(fb_matches):
                letter = m_info[0].upper()
                if letter != 'A' and letter != '1':
                    continue
                curr_seq = [m_info]
                curr_letter = letter
                for next_m in fb_matches[i+1:]:
                    next_letter = next_m[0].upper()
                    if next_letter > curr_letter:
                        letter_gap = ord(next_letter) - ord(curr_letter)
                        if letter_gap <= 2:
                            curr_seq.append(next_m)
                            curr_letter = next_letter
                if len(curr_seq) > len(fb_best):
                    fb_best = curr_seq
            if len(fb_best) >= 2:
                stem = content[:fb_best[0][1].start()].strip()
                for i, m_info in enumerate(fb_best):
                    opt_letter = m_info[0]
                    m = m_info[1]
                    start_idx = m.end()
                    end_idx = fb_best[i+1][1].start() if i + 1 < len(fb_best) else len(content)
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
        opt_line_pattern = re.compile(r'^\s*(?:\()?([A-Ha-h])(?:\)|\.|:-)\s*(.*)', re.IGNORECASE)
        first_opt_idx = -1
        found_opts = {}
        for idx, line in enumerate(lines):
            # If an explicit Answer / Solution marker is encountered, stop collecting options
            if re.match(r'^\s*(?:\*\*\s*(?:Answer|Ans(?:wer)?|Correct(?:\s*Answer)?|Solution|Explanation|ഉത്തരം|വിശദീകരണം)\s*\*\*|\*?\*?(?:Answer|Ans(?:wer)?|Correct(?:\s*Answer)?|Solution|Explanation|ഉത്തരം|വിശദീകരണം)\*?\*?\s*(?:[\:\.\-]+|\s*$))', line, re.IGNORECASE):
                trailing_block = '\n'.join(lines[idx:])
                _, ans, exp = parse_trailing_meta(trailing_block)
                if ans and not marked_answer:
                    marked_answer = ans
                if exp and not explanation:
                    explanation = exp
                break
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
                found_opts[m_line.group(1)] = clean_opt
        if len(found_opts) >= 2:
            opt_keys = [k.upper() for k in found_opts.keys()]
            if opt_keys[0] in ('A', '1') and (len(opt_keys) == 1 or ord(opt_keys[1]) - ord(opt_keys[0]) <= 2):
                options = found_opts
                stem = '\n'.join(lines[:first_opt_idx]).strip()
    
    # Check if marked answer or explanation was in full content or stem
    if not marked_answer or not explanation:
        _, c_ans, c_exp = parse_trailing_meta(content)
        if c_ans and not marked_answer:
            marked_answer = c_ans
        if c_exp and not explanation:
            explanation = c_exp

    if not marked_answer or not explanation:
        clean_stem, s_ans, s_exp = parse_trailing_meta(stem)
        if s_ans and not marked_answer:
            marked_answer = s_ans
            stem = clean_stem
        if s_exp and not explanation:
            explanation = s_exp
            stem = clean_stem

    # Strip marks indicators (e.g. '[1 mark]', '[2 marks]', '(1 mark)') from stem
    stem = re.sub(r'(?i)\s*\[\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\]', '', stem)
    stem = re.sub(r'(?i)\s*\(\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\)', '', stem)
    stem = re.sub(r'\n{3,}', '\n\n', stem).strip()

    from ..ingest.option_answer_separator import separate_options_dict, heal_assertion_reason_question
    options, marked_answer, explanation = separate_options_dict(options, marked_answer, explanation)
    stem, options = heal_assertion_reason_question(stem, options)

    return stem, options, marked_answer, explanation


def classify_question_type(options: dict, answer_text: str = None, marked_answer: str = None, stem: str = None) -> tuple[str, Optional[dict]]:
    """
    Determines if a question is MCQ or descriptive.
    """
    from ..ingest.option_answer_separator import classify_question_type as _classify_question_type
    return _classify_question_type(options, answer_text, marked_answer=marked_answer, stem=stem)


def is_instruction_or_header(text: str) -> bool:
    """Detects if a text snippet is exam instructions or document header rather than a question."""
    if not text:
        return False
    t = text.strip()[:250].lower()
    instruction_starters = (
        'all questions are',
        'each question carries',
        'the question paper contains',
        'the question paper consists',
        'general instructions',
        'instructions for candidate',
        'instructions to candidate',
        'candidates are required',
        'candidate should',
        'time allowed',
        'maximum marks',
        'total marks',
        'marks are indicated',
        'marks for each',
        'read the following instructions',
        'do not use calculator',
        'use of calculator',
        'sections in the question paper',
        'there are 3 sections',
        'there are 4 sections',
        'provisional answer key',
        'question paper code',
        'category code',
    )
    return any(t.startswith(s) or f" {s}" in t or f"\n{s}" in t for s in instruction_starters)


def is_equation_or_calculation(text: str) -> bool:
    """Detects if text is purely an equation or calculation remnant (no natural language question)."""
    if not text:
        return True
    clean = text.strip()
    # Remove math tags / LaTeX / markdown / numbers / symbols
    stripped = re.sub(r'\{\{EQ:.*?\}\}', 'EQUATION', clean)
    stripped = re.sub(r'\$[^$]*\$', '', stripped)
    stripped = re.sub(r'<[^>]+>', '', stripped)
    stripped = re.sub(r'[\d\s\=\+\-\*\/\^\_\(\)\[\]\{\}\\\.\,\:\;\%\>\<\~\|\#\±\×\÷\√\∮\∴\⇒\σ\ε\ϕ\π\μ\τ]', '', stripped)
    # If fewer than 4 alphabetic / Indic characters remain, it's not a real question
    return len(stripped) < 4


def looks_like_question(stem: str, options: dict = None, marked_answer: str = None, sub_parts: dict = None) -> bool:
    """Verifies that a block possesses genuine question characteristics."""
    check_text = (stem or "").strip()
    if not check_text and sub_parts and isinstance(sub_parts, dict):
        check_text = "\n".join(str(v) for v in sub_parts.values()).strip()
    if not check_text and options and isinstance(options, dict):
        check_text = "\n".join(str(v) for v in options.values()).strip()

    if not check_text:
        return False
    s = check_text.lower()
    if is_instruction_or_header(s):
        return False

    # Positive confirmations: these unconditionally qualify a question block
    if options and len(options) >= 2:
        return True
    if sub_parts and len(sub_parts) >= 1:
        return True
    if marked_answer:
        return True
    if '{{IMG' in check_text or '{{EQ:' in check_text or '<svg' in check_text or 'diagram' in s:
        return True
    if '?' in check_text or '؟' in check_text:
        return True

    q_words = (
        'what', 'which', 'why', 'how', 'when', 'where', 'who', 'whom',
        'calculate', 'find', 'determine', 'explain', 'describe', 'state',
        'prove', 'show', 'derive', 'write', 'define', 'evaluate', 'solve',
        'assertion', 'reason', 'consider', 'choose', 'match', 'list', 'differentiate',
        'distinguish', 'obtain', 'verify', 'establish', 'mention', 'name', 'identify',
        'give', 'express', 'compute', 'draw', 'illustrate',
        # Physics / Science keywords that confirm a question context
        'torque', 'flux', 'charge', 'field', 'dipole', 'shell', 'radius', 'surface',
        'magnitude', 'intensity', 'density', 'electric', 'magnetic', 'force', 'potential',
        'placed', 'parallel', 'spherical', 'cylindrical', 'hollow', 'conducting',
        'inclined', 'uniform', 'infinite', 'metallic', 'linked', 'enclosed',
        'എത്ര', 'ഏത്', 'ഏവ', 'എങ്ങനെ', 'എന്ത്', 'ആര്', 'എവിടെ',
        'കണ്ടെത്തുക', 'വിശദീകരിക്കുക', 'പ്രസ്താവന', 'ശരിയായത്', 'തെറ്റായത്',
        'എഴുതുക', 'നിർവചിക്കുക', 'തെളിയിക്കുക', 'കണക്കാക്കുക', 'ചേരുംപടി'
    )
    if any(re.search(rf'\b{re.escape(w)}', s, re.IGNORECASE) for w in q_words):
        return True
    if re.search(r'\[\s*\d+\s*(?:marks?|mark|m)\s*\]|\(\s*\d+\s*(?:marks?|mark|m)\s*\)', s):
        return True

    # For text-only blocks without options, subparts, or question words, verify not just an equation artifact
    if is_equation_or_calculation(check_text):
        return False

    clean_end = s.rstrip()
    if not re.search(r'(?::|,|;|and|or|of|to|in|at|from|ratio|between|with|is|are)\s*$', clean_end):
        return len(s) >= 3
    return len(s) > 10


def find_sequential_question_boundaries(candidates: list[dict], gap_tolerance: int = 25) -> list[dict]:
    """
    Scans candidates in document order and keeps only a strictly forward-moving
    sequence of question boundaries.

    Guarantees:
    - Never reconstructs after the fact with speculative heuristic chains.
    - Selects the true starting question (Q1) if preceded by table/code numbers.
    - Follows an expected counter: accepting q == expected, and forward jumps up to +gap_tolerance.
    - Protects against numbered sub-lists inside explanations/stems from triggering false question splits.
    - Handles deliberate section restarts (e.g. 'Section B \n 1.') seamlessly.
    """
    if not candidates:
        return []

    # Choose optimal starting candidate: prefer candidate with q_num == 1 near document start
    start_idx = 0
    if candidates[0]['q_num'] != 1:
        for idx, c in enumerate(candidates[:6]):
            if c['q_num'] == 1 and (c.get('is_explicit') or (idx + 1 < len(candidates) and candidates[idx+1]['q_num'] in (2, 3))):
                start_idx = idx
                break

    accepted = [candidates[start_idx]]
    expected = candidates[start_idx]['q_num'] + 1

    for idx, c in enumerate(candidates[start_idx + 1:], start=start_idx + 1):
        q = c['q_num']
        is_explicit = c.get('is_explicit', False)
        is_sec_restart = c.get('is_section_restart', False)

        if q == expected:
            accepted.append(c)
            expected = q + 1
        elif (is_explicit or is_sec_restart) and q <= expected - 1:
            # Deliberate section restart (e.g. Part B restarting at 1, or explicit 'Question 1:')
            accepted.append(c)
            expected = q + 1
        elif q > expected:
            # Check lookahead to prevent false positives (like numbers in formulas) from 
            # swallowing actual sequential questions. Only jump if expected doesn't appear soon.
            lookahead_qnums = [cand['q_num'] for cand in candidates[idx+1:idx+20]]
            if expected in lookahead_qnums or (expected + 1) in lookahead_qnums or (expected + 2) in lookahead_qnums:
                continue
                
            accepted.append(c)
            expected = q + 1

    return accepted


def parse_questions_from_text(raw_text: str, token_tracker: dict = None) -> list[dict]:
    """Extracts question blocks from raw text using regex with LLM fallback."""
    if not raw_text or not raw_text.strip():
        return []

    bad_prev_words = {
        'section', 'part', 'class', 'unit', 'chapter', 'table', 'figure', 'fig',
        'date', 'time', 'marks', 'mark', 'page', 'code', 'no', 'sl', 'step', 'case',
        'ratio', 'between', 'and', 'or', 'of', 'to', 'in', 'at', 'from', 'with', 'is', 'are', 'was', 'were', 'by'
    }

    # Unified question matching:
    # 1. Explicit markers: Question, Q, Q.No., Sl.No., Problem, Prob., ചോദ്യം, പ്രശ്നം
    explicit_pat = re.compile(
        r'(?:^|\n)\s*[\*\_]*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)[\*\_]*(\d{1,3})\s*[\*\_]*(?::\-?|[\.\):-])\s*[\*\_]*(?=\s+[^\=\+\-\*\/]|\s*$)',
        re.IGNORECASE
    )
    # 2. Parenthesized or bracketed numbers at line start: (1) ... or [1] ...
    paren_pat = re.compile(
        r'(?:^|\n)\s*[\*\_]*[\(\[]\s*(\d{1,3})\s*[\)\]]\s*[\.\:\-]?\s*[\*\_]*(?=[^\d\s\=\+\-\*\/]|\s+[^\=\+\-\*\/])',
        re.IGNORECASE
    )
    # 3. Bare numbers at line start: 1. or 1) or 1:-
    num_pat = re.compile(
        r'(?:^|\n)([ \t]*)[\*\_]*(\d{1,3})[\*\_]*\s*(?:[\.\)]|:-)\s*[\*\_]*(?=[^\d\s\=\+\-\*\/]|\s+[^\=\+\-\*\/])',
        re.IGNORECASE
    )

    all_raw_matches = []
    # Collect explicit matches
    for m in explicit_pat.finditer(raw_text):
        all_raw_matches.append((m.start(), m.end(), int(m.group(1)), m, True))

    # Collect parenthesized matches
    for m in paren_pat.finditer(raw_text):
        all_raw_matches.append((m.start(), m.end(), int(m.group(1)), m, False))

    # Collect bare number matches
    for m in num_pat.finditer(raw_text):
        all_raw_matches.append((m.start(), m.end(), int(m.group(2)), m, False))

    # Deduplicate overlapping matches by start position, prioritizing explicit matches
    all_raw_matches.sort(key=lambda x: (x[0], -1 if x[4] else 0))
    deduped_matches = []
    occupied_spans = []
    for start, end, q_num, m, is_exp in all_raw_matches:
        overlap = any(not (end <= s or start >= e) for s, e in occupied_spans)
        if not overlap:
            occupied_spans.append((start, end))
            deduped_matches.append((start, end, q_num, m, is_exp))

    # Filter candidate matches against invalid preceding words / punctuation
    #
    # NOTE: the "does the preceding text end with trailing-off punctuation"
    # checks below (':', '-', '/', '>', etc.) were written to catch a number
    # that's really part of an equation or a hyphenated fragment (e.g.
    # "...ratio 4-" should not become "Question 4"). But the inline equation
    # /diagram images this pipeline generates for numeric answers are HTML
    # tags like {{IMG:https://...}}, and that
    # markup's own closing "/>" is made of exactly the same characters
    # ('/' and '>') the filter is watching for. A REAL question marker that
    # happens to follow an equation image in the raw text — e.g. Q3's answer
    # "5 x 10^5 V/m" rendered as an inline image, immediately followed by
    # "4. In the given figure..." — gets its trailing "/>" mistaken for
    # trailing-off prose and the genuine Q4 marker is silently rejected as a
    # candidate. Its entire content then gets absorbed into the PRECEDING
    # question's block instead of becoming its own question. Strip a
    # trailing HTML tag remnant before applying the punctuation check so the
    # markup's own syntax can't be confused with real prose punctuation.
    candidates = []
    prev_end = 0
    for start, end, q_num, m, is_exp in deduped_matches:
        if q_num < 1 or q_num > 300:
            continue

        # Check intervening text between previous candidate's end and this one for section headers
        intervening = raw_text[prev_end:start]
        sec_match = re.search(r'(?i)\b(?:section|part|group|വിഭാഗം|ഖണ്ഡിക)\s*[-:]?\s*([A-Za-z0-9]+)', intervening)
        is_sec_restart = bool(sec_match)
        sec_name = sec_match.group(0).strip() if sec_match else None
        
        if not is_sec_restart:
            ans_key_match = re.search(r'(?i)\b(?:answer\s*key|answers|solutions|ഉത്തരങ്ങൾ|പരിഹാരങ്ങൾ)', intervening)
            if ans_key_match:
                is_sec_restart = True
                sec_name = "Answer Key"

        if not is_exp:
            # Clean preceding text of all HTML comments (<!--...-->) and tags (<...>)
            preceding_raw = raw_text[max(0, start - 150):start]
            preceding_clean = re.sub(r'<!--.*?-->', '', preceding_raw, flags=re.DOTALL)
            preceding_clean = re.sub(r'<[^>]+>', '', preceding_clean)
            # Inline equation/diagram anchors are pipeline markup, not prose
            # continuation. Remove them before checking the previous text.
            preceding_clean = re.sub(r'\{\{(?:IMG|EQ):.*?\}\}', '', preceding_clean, flags=re.DOTALL)

            # Check if match begins on a new line (starts with \n, or start is at 0, or preceding char was \n)
            matched_str = m.group(0)
            is_new_line = matched_str.startswith('\n') or start == 0 or (start > 0 and raw_text[start - 1] == '\n')

            if is_new_line:
                # Candidate starts on a new line!
                # Examine the preceding non-empty line
                lines = [l.strip() for l in preceding_clean.split('\n') if l.strip()]
                if lines:
                    prev_line = lines[-1]
                    # Check 1: Incomplete math operators or word hyphenation at end of previous line
                    if prev_line.endswith(('=', '+', '*', '/', '^')) or re.search(r'[a-zA-Z]-$', prev_line):
                        continue
                    # Check 2: Trailing ratio / math continuation (e.g. "...in the ratio 2 : \n 3.")
                    if re.search(r'\b(?:ratio|between|equals?|equal\s+to)\s*(?:of\s*)?(?:\d+\s*)?[:=]?\s*$', prev_line, re.IGNORECASE):
                        continue
            else:
                # Candidate has text on the SAME line preceding it (e.g. "Figure 1", "Table 2", "ratio 2 : 3")
                same_line_prefix = preceding_clean.split('\n')[-1].strip()
                if same_line_prefix:
                    same_line_words = same_line_prefix.split()
                    last_word = same_line_words[-1].rstrip('.:-').lower() if same_line_words else ''
                    if last_word in bad_prev_words:
                        continue
                    if same_line_prefix.endswith((':', ',', ';', '-', '–', '—', '=', '+', '*', '/', '^', '>', '<')):
                        continue

        snippet = raw_text[end:end + 200].strip()
        if is_instruction_or_header(snippet):
            continue

        candidates.append({
            'match': m,
            'q_num': q_num,
            'start': start,
            'end': end,
            'is_explicit': is_exp,
            'is_section_restart': is_sec_restart,
            'section_name': sec_name,
        })
        prev_end = end

    candidates.sort(key=lambda x: x['start'])
    valid_candidates = find_sequential_question_boundaries(candidates) if candidates else []
    if not valid_candidates and candidates:
        valid_candidates = candidates

    blocks = []
    current_section = None
    if valid_candidates:
        for i, c in enumerate(valid_candidates):
            if c.get("section_name"):
                current_section = c["section_name"]
            
            start_idx = c['end']
            end_idx = valid_candidates[i+1]['start'] if i + 1 < len(valid_candidates) else len(raw_text)
            block_content = raw_text[start_idx:end_idx].strip()
            if block_content:
                page_markers = list(re.finditer(r'<!--QAI_PAGE:(\d+)-->', raw_text[:c["start"]]))
                blocks.append({
                    "q_num": c["q_num"],
                    "content": block_content,
                    "page_number": int(page_markers[-1].group(1)) if page_markers else None,
                    "section_name": current_section,
                })

    extracted = []
    for block_data in blocks:
        q_num = block_data["q_num"]
        content = block_data["content"]
        page_number = block_data.get("page_number")
        section_name = block_data.get("section_name")

        stem, options, marked_answer, explanation = separate_stem_and_options(content)
        question_type, sub_parts = classify_question_type(options, explanation, marked_answer=marked_answer, stem=stem)

        # If sub_parts not found from options, try extracting from stem (for descriptive questions)
        if not sub_parts and stem and question_type != "mcq":
            from ..ingest.option_answer_separator import extract_subparts_from_text, is_valid_subparts
            extracted_sub = extract_subparts_from_text(stem)
            if extracted_sub and is_valid_subparts(extracted_sub):
                sub_parts = extracted_sub
                question_type = "descriptive_subquestions"

        # If sub_parts exists and is valid, extract only the introductory stem (strip sub-questions out of stem)
        # to guarantee ZERO DUPLICATION of question text in both stem and sub_parts!
        if sub_parts and stem:
            first_key = sorted(sub_parts.keys())[0]
            fk_esc = re.escape(first_key)
            first_sub_pattern = re.compile(
                rf'(?:(?:^|\n|\s{{2,}}|\.\s+)\([{fk_esc}{fk_esc.upper()}]\)\s*)|'
                rf'(?:(?:^|\n|[ \t]{{2,}}|\.\s+)[{fk_esc}{fk_esc.upper()}][\.\)\:-]\s*)'
            )
            m_sub = first_sub_pattern.search(stem)
            if m_sub:
                intro = stem[:m_sub.start()].strip()
                if re.match(r"^(?:question\s*\d*|q\d*|ചോദ്യം\s*\d*|[\d\.\:\-\s])*$", intro, re.IGNORECASE):
                    stem = ""
                else:
                    stem = intro

        # Build answer_text from explanation or from the answer section
        answer_text = explanation
        q_options = {k.upper(): v for k, v in options.items()} if question_type == "mcq" else {}

        # Validate that stem/block looks like a genuine question (with safety net for accepted boundaries)
        looks_ok = looks_like_question(stem, q_options, marked_answer, sub_parts=sub_parts)
        if not looks_ok and len(content.strip()) >= 3 and not is_instruction_or_header(content):
            looks_ok = True

        if looks_ok:
            if section_name and re.search(r'(?i)answer\s*key|answers|solutions|ഉത്തരങ്ങൾ|പരിഹാരങ്ങൾ', section_name):
                target_q = next((q for q in extracted if q["question_number"] == q_num and q.get("section") != section_name), None)
                if target_q:
                    if marked_answer:
                        target_q["marked_answer"] = marked_answer
                        target_q["marked_correct_answer"] = marked_answer
                    
                    ans_text = stem if stem else explanation
                    if ans_text:
                        clean_ans = ans_text.strip()
                        if len(clean_ans) == 1 and clean_ans.upper() in "ABCDE1234":
                            target_q["marked_answer"] = clean_ans.upper()
                            target_q["marked_correct_answer"] = clean_ans.upper()
                        elif not target_q.get("answer_text"):
                            target_q["answer_text"] = ans_text
                        else:
                            target_q["answer_text"] += "\n" + ans_text
                    continue
            
            extracted.append({
                "question_number": q_num,
                "stem": stem,
                "options": q_options,
                "question_type": question_type,
                "sub_parts": sub_parts,
                "answer_text": answer_text,
                "marked_answer": marked_answer if question_type == "mcq" else None,
                "marked_correct_answer": marked_answer if question_type == "mcq" else None,
                "explanation": explanation,
                "page_number": page_number,
                "section": section_name,
            })

    # If duplicate question numbers exist (e.g. Section A: 1..5 and Section B: 1..5),
    # renumber sequentially to guarantee unique IDs downstream while saving original_question_number
    seen_qnums = set()
    has_duplicate_qnums = any(q["question_number"] in seen_qnums or seen_qnums.add(q["question_number"]) for q in extracted)
    if has_duplicate_qnums:
        for idx, q in enumerate(extracted, start=1):
            q["original_question_number"] = q["question_number"]
            q["question_number"] = idx
    print(f"Regex parser extracted {len(extracted)} questions.", flush=True)
    return extracted or []


def extract_node(state: GraphState) -> GraphState:
    """
    Extracts question blocks from the PDF using the layout-aware pipeline
    when span_blocks are available, or the existing regex/LLM path otherwise.

    Layout-aware path (PDFs with a live text layer):
        1. segment_questions(span_blocks) -- structural boundary detection via
           font size, bold flag, and x-indent (not pattern matching on flat text)
        2. classify_block_with_llm(segment, llm) per segment -- schema-constrained
           LLM call that reads layout hints and handles "A.C. current" correctly
        3. Falls back to separate_stem_and_options on any per-segment LLM failure

    Regex/LLM fallback path (scanned/OCR-only PDFs, or when span_blocks is empty):
        parse_questions_from_text(raw_text) -- unchanged from the previous pipeline
    """
    raw_text = state.get("raw_text", "")
    span_blocks = state.get("span_blocks") or []

    if "token_usage" not in state:
        state["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if span_blocks:
        print(f"[extract_node] Layout-aware path: {len(span_blocks)} span elements.", flush=True)
        try:
            from ..ingest.layout_segmenter import segment_questions
            from ..ingest.block_classifier import classify_block_with_llm

            segments = segment_questions(span_blocks)
            print(f"[extract_node] Segmenter found {len(segments)} question segments.", flush=True)

            if segments:
                import pymupdf
                from ..ingest.structural_ocr import perform_structural_ocr
                from ..ingest.quality_check import is_question_unreadable
                
                doc_path = state.get("document_id") or state.get("file_path")
                doc = pymupdf.open(doc_path) if doc_path else None
                open_pages = {}
                
                extracted = []
                
                # Step 10 constraint: Max N calls per doc
                structural_ocr_calls = 0
                max_ocr_calls = int(os.getenv("MAX_STRUCTURAL_OCR_CALLS", "100"))
                
                enable_structural_ocr = (
                    is_gemini_enabled()
                    and os.getenv("ENABLE_MATH_OCR", "false").lower() in ("true", "1")
                )
                for seg in segments:
                    had_gemini_ocr_call = False
                    # Perform Structural OCR on flagged elements if enabled (Math OCR in equations)
                    if doc and enable_structural_ocr:
                        for elem in seg.elements:
                            if getattr(elem, "needs_structural_ocr", False):
                                if structural_ocr_calls < max_ocr_calls:
                                    p_num = elem.page
                                    if p_num not in open_pages:
                                        open_pages[p_num] = doc[p_num]
                                        
                                    # Apply padding
                                    b = elem.bbox
                                    padded_bbox = [max(0, b[0] - 5), max(0, b[1] - 5), b[2] + 5, b[3] + 5]
                                    try:
                                        pix = open_pages[p_num].get_pixmap(clip=padded_bbox, dpi=200)
                                        img_bytes = pix.tobytes("png")
                                        latex_out, ambiguous, needs_review, p_tok, c_tok = perform_structural_ocr(
                                            img_bytes, elem.region_type
                                        )
                                        state["token_usage"]["prompt_tokens"] += p_tok
                                        state["token_usage"]["completion_tokens"] += c_tok
                                        state["token_usage"]["total_tokens"] += (p_tok + c_tok)
                                        structural_ocr_calls += 2 # since it makes 2 calls
                                        had_gemini_ocr_call = True
                                        
                                        elem.text = f"{{{{EQ:{latex_out}}}}}"
                                        if needs_review:
                                            elem.ocr_uncertain = True
                                            
                                    except Exception as e:
                                        print(f"Structural OCR failed: {e}")
                                        elem.ocr_uncertain = True
                                else:
                                    # Cap exceeded: DO NOT silently skip; flag for admin review queue
                                    elem.ocr_uncertain = True
                                    elem.ocr_limit_exceeded = True
                                    print(f"[extract_node] OCR cap ({max_ocr_calls}) reached. Flagging element for review queue.")
                                    
                    llm = _init_llm(state)
                    q = classify_block_with_llm(seg, llm=llm)
                    if q:
                        if seg.section and re.search(r'(?i)answer\s*key|answers|solutions|ഉത്തരങ്ങൾ|പരിഹാരങ്ങൾ|hints', seg.section):
                            target_q = next((t for t in extracted if t["question_number"] == seg.q_num and t.get("section") != seg.section), None)
                            if target_q:
                                ans_text = q.get("stem", "")
                                if not ans_text:
                                    ans_text = q.get("explanation", "")
                                if ans_text:
                                    clean_ans = ans_text.strip()
                                    if len(clean_ans) == 1 and clean_ans.upper() in "ABCDE1234":
                                        target_q["marked_answer"] = clean_ans.upper()
                                        target_q["marked_correct_answer"] = clean_ans.upper()
                                    elif not target_q.get("answer_text"):
                                        target_q["answer_text"] = ans_text
                                    else:
                                        target_q["answer_text"] += "\n" + ans_text
                                continue

                        q["question_number"] = seg.q_num
                        if "stem" in q and q["stem"]:
                            q["stem"] = re.sub(rf'^(?:Question\s*(?:No\.?)?\s*|Q\.?\s*(?:No\.?)?\s*|Sl\.?\s*No\.?\s*|Problem\s*|Prob\.?\s*)?{seg.q_num}\s*(?:[\.\)\:-]|--)\s*', '', q["stem"], flags=re.IGNORECASE)
                        q.setdefault("page_number", seg.page)
                        q.setdefault("section", seg.section)
                        # Ensure required keys are present for downstream nodes
                        q.setdefault("question_type", "descriptive_single")
                        q.setdefault("sub_parts", None)
                        q.setdefault("options", {})
                        q.setdefault("marked_answer", None)
                        q.setdefault("marked_correct_answer", None)
                        q.setdefault("explanation", None)
                        q.setdefault("answer_text", q.get("explanation"))

                        # Flag unreadable/garbled questions without making automatic AI Vision calls
                        if is_question_unreadable(q.get("stem", ""), q.get("options", {})):
                            q["ocr_ambiguous"] = True
                            q["is_garbled"] = True
                        
                        # Pass review flag to the parsed question if any element triggered it
                        has_uncertain = any(getattr(e, "ocr_uncertain", False) for e in seg.elements)
                        has_cap_exceeded = any(getattr(e, "ocr_limit_exceeded", False) for e in seg.elements)
                        
                        if has_uncertain or has_cap_exceeded:
                            q["ocr_ambiguous"] = True
                            flag_type = "ocr_limit_exceeded" if has_cap_exceeded else "equation_ocr_uncertain"
                            explanation = (
                                f"Structural OCR budget exceeded ({max_ocr_calls} calls). Equation was not processed via Vision OCR."
                                if has_cap_exceeded else
                                "Structural OCR detected ambiguity or low confidence"
                            )
                            
                            # Log to ReviewFlag
                            try:
                                from ..storage.database import SessionLocal
                                from ..storage.models import ReviewFlag
                                
                                db = SessionLocal()
                                db_flag = ReviewFlag(
                                    job_id=doc_path, # doc_path acts as job_id here
                                    question_number=q["question_number"],
                                    flag_type=flag_type,
                                    original_text="",
                                    proposed_text="",
                                    llm_explanation=explanation,
                                    confidence=0.5
                                )
                                db.add(db_flag)
                                db.commit()
                                db.close()
                            except Exception as e:
                                print(f"Failed to save ReviewFlag: {e}")
                            
                        extracted.append(q)

                    # Only throttle briefly if a Gemini OCR call was made
                    if had_gemini_ocr_call:
                        time.sleep(1.0)

                if doc:
                    doc.close()

                # De-duplicate and re-number if section restarts caused duplicates
                seen_qnums: set = set()
                has_dup = any(
                    q["question_number"] in seen_qnums
                    or seen_qnums.add(q["question_number"])  # type: ignore[func-returns-value]
                    for q in extracted
                )
                if has_dup:
                    for idx, q in enumerate(extracted, start=1):
                        q["original_question_number"] = q["question_number"]
                        q["question_number"] = idx

                print(f"[extract_node] Layout-aware extraction produced {len(extracted)} questions.", flush=True)
                
                # Strip leading question numbers from stems
                prefix_re = re.compile(r'^[Qq]*[\.\s]*\d+[\.\)\-:]?\s*')
                for q in extracted:
                    if q.get("stem"):
                        q["stem"] = prefix_re.sub('', q["stem"])
                
                return {"extracted_blocks": extracted, "token_usage": state.get("token_usage", {})}

            # Segmenter found no boundaries — fall through to regex path
            print("[extract_node] Segmenter found 0 segments, falling back to regex path.", flush=True)

        except Exception as e:
            print(f"[extract_node] Layout-aware path failed ({e}), falling back to regex path.", flush=True)
            import traceback as _tb
            _tb.print_exc()

    # --- Existing regex / LLM fallback path (unchanged) ---
    print("[extract_node] Using regex/LLM path.", flush=True)
    extracted = parse_questions_from_text(raw_text, state["token_usage"])
    
    # Strip leading question numbers from stems
    prefix_re = re.compile(r'^[Qq]*[\.\s]*\d+[\.\)\-:]?\s*')
    for q in extracted:
        if q.get("stem"):
            q["stem"] = prefix_re.sub('', q["stem"])

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
    """Filters out malformed questions, invalid numbers (e.g. Q0), non-questions, and dedups."""
    blocks = state.get("extracted_blocks", [])
    processed = state.get("processed_questions", [])

    def _filter_and_dedup(items: list[dict]) -> list[dict]:
        valid_items = []
        seen = {}
        for b in items:
            if not isinstance(b, dict):
                continue
            q_num = b.get("question_number")
            if q_num is None or q_num < 1:
                continue
            stem = b.get("stem", "")
            options = b.get("options", {})
            marked_ans = b.get("marked_correct_answer") or b.get("marked_answer")

            sub_parts = b.get("sub_parts")
            # Check if this item is a genuine question
            if not looks_like_question(stem, options, marked_ans, sub_parts=sub_parts):
                if len(stem.strip()) < 3:
                    continue

            # Standardize marked_answer
            if isinstance(options, dict) and options and marked_ans:
                if marked_ans not in options:
                    for k in options:
                        if str(k).upper() == str(marked_ans).upper():
                            b["marked_answer"] = k
                            if "marked_correct_answer" in b:
                                b["marked_correct_answer"] = k
                            break

            # Deduplication: ONLY deduplicate if the item is a true duplicate (identical or near-identical stem)
            if q_num in seen:
                prev_idx = seen[q_num]
                prev_b = valid_items[prev_idx]
                prev_stem_clean = re.sub(r'\s+', ' ', prev_b.get("stem", "")).strip().lower()
                curr_stem_clean = re.sub(r'\s+', ' ', stem).strip().lower()
                is_same_stem = (prev_stem_clean == curr_stem_clean or 
                                (len(prev_stem_clean) > 20 and prev_stem_clean[:50] == curr_stem_clean[:50]))
                
                if is_same_stem:
                    # True duplicate: prefer version with options or longer stem
                    prev_has_opts = bool(prev_b.get("options"))
                    curr_has_opts = bool(options)
                    if curr_has_opts and not prev_has_opts:
                        valid_items[prev_idx] = b
                    elif len(stem) > len(prev_b.get("stem", "")):
                        valid_items[prev_idx] = b
                else:
                    # Different stems sharing a question number: do NOT overwrite! Append as separate question!
                    valid_items.append(b)
            else:
                seen[q_num] = len(valid_items)
                valid_items.append(b)

        return valid_items

    valid_blocks = _filter_and_dedup(blocks)
    valid_processed = _filter_and_dedup(processed) if processed else []

    # Safeguard: If valid_processed is empty but valid_blocks has items, fallback to valid_blocks
    if not valid_processed and valid_blocks:
        valid_processed = [dict(b) for b in valid_blocks]

    # Keep summary in sync
    summary = state.get("summary", {})
    if summary:
        summary["total"] = len(valid_processed) if valid_processed else len(valid_blocks)

    print(f"Rule check validated {len(valid_blocks)} extracted blocks and {len(valid_processed)} processed questions.", flush=True)
    return {
        "extracted_blocks": valid_blocks,
        "processed_questions": valid_processed,
        "summary": summary
    }


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
            engine = get_setting("question_ocr_engine", "gemini")
    except Exception as err:
        if "disabled in Pipeline Settings" in str(err):
            raise
    engine = str(engine or "gemini").strip().lower()
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

            question_type, sub_parts = classify_question_type(options or {}, explanation)
            q_options = (options or {}) if question_type == "mcq" else {}

            data = {
                "question_number": question_number,
                "stem": stem or raw_text,
                "options": q_options,
                "question_type": question_type,
                "sub_parts": sub_parts,
                "answer_text": explanation,
                "marked_answer": marked_ans,
                "marked_correct_answer": marked_ans,
                "explanation": explanation
            }
            return data, 0, 0
        except Exception as e:
            print(f"[Question OCR] Tesseract OCR failed: {e}", flush=True)
            raise

    prompt = """
Extract ONLY Question {question_number} from this page exactly as it appears for STUDENTS studying for exams.
The text must make complete, clear pedagogical and grammatical sense to a student.
CRITICAL INSTRUCTIONS:
1. Transcribe and preserve all mathematical equations, fractions, formulas, and symbols into standard inline LaTeX ($...) or display LaTeX ($$...$$).
   - DO NOT put regular English or Malayalam sentences, or subquestion labels (like "(a)" or "(b)") inside math delimiters ($...$ or $$...$$). Words inside math mode lose spaces and become unreadable.
   - WRONG: `$\text{(b) The reactance is given by } X_C = \frac{1}{\omega C}$`
   - CORRECT: `(b) The reactance is given by $X_C = \frac{1}{\omega C}$`
   - Explanatory phrases must remain normal text outside math mode: write "where $h$ is Planck's constant and $e$ is the charge of an electron", never "whereh $$isPlanck'sconstantande$$".
   - Keep spaces between words and variables (write "where $h$", not "whereh").
2. Preserve all image tokens (e.g., {{IMG:https://...}} or {{IMG_BLOCK:...}}) EXACTLY as they appear in the text. NEVER remove or modify them.
3. Determine if the question is "mcq" (Multiple Choice) or "descriptive".
4. For "mcq" questions:
   - Extract options into 'options': {"A": "...", "B": "...", "C": "...", "D": "..."}.
   - Extract the correct answer into 'marked_answer'.
   - Extract any solution or explanation into 'explanation'.
5. For "descriptive" questions (no A/B/C/D choices, often with sub-questions like (a), (b)):
   - Keep the entire question, including sub-questions, inside the 'stem'.
   - Extract the provided solution, answer, or diagrams into 'answer_text'. Make sure to include any {{IMG:...}} tokens that are part of the answer!
   - Leave 'options' empty.
6. Transcribe Malayalam and Indic text faithfully into standard Unicode script. Do not invent or summarize.

Output ONLY a valid JSON object exactly matching this schema, with no markdown code blocks or extra text:
{
  "question_number": {question_number},
  "question_type": "mcq",
  "stem": "Question stem text here\\na. $\\chi^2$ test\\nb. 't' test",
  "options": {
    "A": "Option A text",
    "B": "Option B text"
  },
  "marked_answer": "B",
  "explanation": "Solution or explanation text here",
  "answer_text": null
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
    explanation = b.get("explanation") or b.get("answer_text") or ""
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

    prompt = (
        SOLVE_PROMPT
        .replace("{question_type}", b.get("question_type", ""))
        .replace("{stem}", stem)
        .replace("{sub_parts}", json.dumps(b.get("sub_parts") or {}))
        .replace("{options}", json.dumps(options))
        .replace("{explanation}", explanation)
    )
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
        corrected_explanation = solve_result.get("corrected_explanation") or explanation
        
        # New parsing for auto-fix and garble flagging
        is_garbled = solve_result.get("is_garbled", False)
        corrected_stem = solve_result.get("corrected_stem", b.get("stem", ""))
        corrected_options = solve_result.get("corrected_options", b.get("options", {}))

        try:
            from ..ingest.option_answer_separator import strip_subquestion_labels_from_answer
            corrected_explanation = strip_subquestion_labels_from_answer(corrected_explanation)
        except Exception:
            pass

        try:
            from ..ingest.math_normalizer import normalize_math_in_text
            import re
            if corrected_explanation:
                corrected_explanation = re.sub(r'<(?!img\b|\/?div\b)[^>]+>', '', corrected_explanation)
                corrected_explanation = normalize_math_in_text(corrected_explanation)
            if reasoning:
                reasoning = re.sub(r'<(?!img\b|\/?div\b)[^>]+>', '', reasoning)
                reasoning = normalize_math_in_text(reasoning)
            if corrected_stem:
                corrected_stem = normalize_math_in_text(corrected_stem)
        except Exception:
            pass
        
        is_valid = (str(validated_answer).strip().upper() == str(marked).strip().upper()) if marked and validated_answer else False
        return {
            **b,
            "stem": corrected_stem,
            "options": corrected_options,
            "is_garbled": is_garbled,
            "marked_correct_answer": marked,
            "marked_answer": marked,
            "validated_answer": validated_answer,
            "is_valid": is_valid,
            "confidence": confidence,
            "reasoning": reasoning,
            "explanation": corrected_explanation,
            "answer_text": corrected_explanation,
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