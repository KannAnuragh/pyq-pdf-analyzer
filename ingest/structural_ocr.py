import json
import os
import re
from google import genai
from google.genai import types
import time

def normalize_latex(latex: str) -> str:
    """Normalizes whitespace and common equivalent LaTeX structures for comparison."""
    if not latex: return ""
    l = latex.replace(" ", "").replace("\\limits", "")
    return l

def call_gemini_ocr(img_bytes: bytes, prompt_extra: str = "") -> tuple[dict, int, int]:
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not google_key:
        return {"latex": "", "confidence": "low"}, 0, 0
    client = genai.Client(api_key=google_key)
    
    # Use Gemini 2.5 Flash as requested
    model_name = "gemini-2.5-flash"
    
    prompt = (
        "Extract the mathematical formula or expression from this image snippet into LaTeX.\n"
        "Rules:\n"
        "- Use standard LaTeX commands (e.g. \\frac{}{}, \\alpha, \\beta, etc.).\n"
        "- Always brace sub/superscripts: _{} and ^{}.\n"
        "- If a part is illegible, wrap it in \\unsure{...}.\n"
        "- Output ONLY valid LaTeX (no $ delimiters, no markdown backticks).\n"
        + prompt_extra
    )
    
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "latex": types.Schema(type=types.Type.STRING),
            "confidence": types.Schema(
                type=types.Type.STRING, 
                enum=["high", "medium", "low"]
            ),
        },
        required=["latex", "confidence"]
    )
    
    res = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.4
        )
    )
    
    p_tok = res.usage_metadata.prompt_token_count if res.usage_metadata else 0
    c_tok = res.usage_metadata.candidates_token_count if res.usage_metadata else 0
    
    try:
        data = json.loads(res.text)
        return data, p_tok, c_tok
    except Exception:
        return {"latex": "", "confidence": "low"}, p_tok, c_tok

def perform_structural_ocr(img_bytes: bytes, region_type: str, max_retries: int = 1) -> tuple[str, bool, bool, int, int]:
    """
    Returns (latex_string, ambiguous, review_flagged, prompt_tokens, completion_tokens).
    """
    p_tot, c_tot = 0, 0
    
    for attempt in range(max_retries + 1):
        prompt_extra = ""
        if attempt > 0:
            if region_type == "fraction":
                prompt_extra = "\nSTRICT REMINDER: This image contains a FRACTION. You MUST use \\frac{}{} in your output!"
            elif region_type == "sub_sup":
                prompt_extra = "\nSTRICT REMINDER: This image contains a subscript or superscript. You MUST use _{} or ^{} in your output!"
                
        # Self-consistency check (2 calls)
        res1, p1, c1 = call_gemini_ocr(img_bytes, prompt_extra)
        res2, p2, c2 = call_gemini_ocr(img_bytes, prompt_extra)
        p_tot += (p1 + p2)
        c_tot += (c1 + c2)
        
        l1 = res1.get("latex", "")
        l2 = res2.get("latex", "")
        
        ambiguous = normalize_latex(l1) != normalize_latex(l2)
        best_res = res1 if res1.get("confidence") == "high" else (res2 if res2.get("confidence") == "high" else res1)
        latex_out = best_res.get("latex", "")
        confidence = best_res.get("confidence", "low")
        
        # Structural cross-check
        struct_fail = False
        if region_type == "fraction" and "\\frac" not in latex_out:
            struct_fail = True
        if region_type == "sub_sup" and "_" not in latex_out and "^" not in latex_out:
            struct_fail = True
            
        if not struct_fail:
            needs_review = ambiguous or (confidence == "low")
            return latex_out, ambiguous, needs_review, p_tot, c_tot
            
    # If we exit the loop, the structural check failed even after retries
    return latex_out, ambiguous, True, p_tot, c_tot

