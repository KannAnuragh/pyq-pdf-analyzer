"""
Centralized Configuration Manager for PDF QA Pipeline.

Provides thread-safe access and persistence for runtime settings:
- Answer validation on/off and provider (Groq / Gemini)
- Malayalam word correction on/off and provider (Groq / Gemini)
- Question quality repair on/off
- Math equation OCR engine (auto, groq, gemini, tesseract, mathpix, off)
- Auto-convert math on upload toggle

Settings are persisted in the SQLite `pipeline_config` table and fall back
to environment variables if not yet configured in the database.
"""

import os
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Cache in memory for fast lookup
_CONFIG_CACHE: Dict[str, Any] = {}
_CACHE_INITIALIZED: bool = False


def _get_env_bool(var_name: str, default: bool) -> bool:
    val = os.getenv(var_name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")


def get_default_settings() -> Dict[str, Any]:
    """Returns base defaults derived from environment variables."""
    # Determine default active provider
    groq_enabled = _get_env_bool("ENABLE_GROQ", False)
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    gemini_enabled = _get_env_bool("ENABLE_GEMINI", False)
    gemini_key = (os.getenv("GOOGLE_API_KEY") or "").strip()

    if groq_enabled and groq_key and not groq_key.startswith("your_"):
        default_provider = "groq"
    elif gemini_enabled and gemini_key and not gemini_key.startswith("your_"):
        default_provider = "gemini"
    else:
        default_provider = "groq"

    # Math OCR engine default
    default_math_engine = os.getenv("MATH_OCR_ENGINE", "auto").strip().lower()
    if default_math_engine not in ("auto", "groq", "gemini", "tesseract", "mathpix", "off"):
        default_math_engine = "auto"

    # Question OCR engine default
    default_question_engine = os.getenv("QUESTION_OCR_ENGINE", default_provider if default_provider in ("groq", "gemini") else "tesseract").strip().lower()
    if default_question_engine not in ("groq", "gemini", "tesseract", "off"):
        default_question_engine = "groq"

    word_corr_enabled = _get_env_bool("ENABLE_WORD_CORRECTION", _get_env_bool("ENABLE_CORRECTION", True))

    corr_provider = os.getenv("CORRECTION_LLM_PROVIDER", default_provider).strip().lower()
    if corr_provider in ("false", "0", "no", "off"):
        corr_provider = "none"
    elif corr_provider in ("true", "1", "yes", "on", "auto"):
        corr_provider = default_provider

    return {
        "validation_enabled": _get_env_bool("ENABLE_ANSWER_VALIDATION", True),
        "validation_provider": os.getenv("SOLVE_LLM_PROVIDER", default_provider).strip().lower(),
        "correction_enabled": word_corr_enabled,
        "word_correction_enabled": word_corr_enabled,
        "correction_provider": corr_provider,
        "repair_enabled": _get_env_bool("ENABLE_REPAIR", True),
        "math_ocr_enabled": _get_env_bool("ENABLE_MATH_OCR", True),
        "math_ocr_engine": default_math_engine,
        "question_ocr_enabled": _get_env_bool("ENABLE_QUESTION_OCR", True),
        "question_ocr_engine": default_question_engine,
        "auto_convert_math": _get_env_bool("AUTO_CONVERT_MATH", True),
        "auto_whole_pdf_ocr_threshold": int(os.getenv("AUTO_WHOLE_PDF_OCR_THRESHOLD", "3")),
        "auto_whole_pdf_ocr_engine": os.getenv("AUTO_WHOLE_PDF_OCR_ENGINE", "auto").strip().lower(),
        "groq_model": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        "groq_vision_model": os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b"),
        "gemini_extract_model": os.getenv("GEMINI_EXTRACT_MODEL", "gemini-3.1-flash-lite"),
        "gemini_solve_model": os.getenv("GEMINI_SOLVE_MODEL", "gemini-3.1-pro-preview"),
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
    }


def get_all_settings() -> Dict[str, Any]:
    """
    Retrieves all pipeline settings, reading from the database and
    overlaying on environment defaults.
    """
    global _CONFIG_CACHE, _CACHE_INITIALIZED
    
    defaults = get_default_settings()
    
    try:
        from .database import SessionLocal
        from .models import PipelineSetting
        
        db = SessionLocal()
        try:
            rows = db.query(PipelineSetting).all()
            db_settings = {}
            for r in rows:
                try:
                    db_settings[r.key] = json.loads(r.value)
                except Exception:
                    db_settings[r.key] = r.value
            
            merged = {**defaults, **db_settings}
            if db_settings.get("groq_api_key"):
                os.environ["GROQ_API_KEY"] = db_settings["groq_api_key"]
            if db_settings.get("google_api_key"):
                os.environ["GOOGLE_API_KEY"] = db_settings["google_api_key"]
            _CONFIG_CACHE = merged
            _CACHE_INITIALIZED = True
            return merged
        finally:
            db.close()
    except Exception as e:
        logger.warning("[ConfigManager] Failed to read settings from DB: %s. Using in-memory/defaults.", e)
        if _CONFIG_CACHE:
            return {**defaults, **_CONFIG_CACHE}
        return defaults


def get_setting(key: str, default: Any = None) -> Any:
    """Gets a single configuration setting."""
    all_cfg = get_all_settings()
    return all_cfg.get(key, default)


def update_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persists one or more configuration updates to SQLite and refreshes in-memory cache.
    """
    global _CONFIG_CACHE
    
    valid_keys = set(get_default_settings().keys()) | {"gemini_api_key"}
    
    try:
        from .database import SessionLocal
        from .models import PipelineSetting
        
        db = SessionLocal()
        try:
            # First, normalize everything into a single dict of final values to write
            final_updates = {}
            for k, v in updates.items():
                if k not in valid_keys:
                    continue
                
                # Normalize types
                if k in ("validation_enabled", "correction_enabled", "word_correction_enabled", "repair_enabled", "auto_convert_math", "math_ocr_enabled", "question_ocr_enabled", "direct_llm_validation"):
                    v = bool(v)
                    final_updates[k] = v
                    if k == "word_correction_enabled":
                        final_updates["correction_enabled"] = v
                    elif k == "correction_enabled":
                        final_updates["word_correction_enabled"] = v
                elif k in ("validation_provider", "correction_provider"):
                    v = str(v).strip().lower()
                    if v not in ("groq", "gemini", "none"):
                        v = "groq"
                    final_updates[k] = v
                elif k == "math_ocr_engine":
                    v = str(v).strip().lower()
                    if v not in ("auto", "groq", "gemini", "tesseract", "mathpix", "off"):
                        v = "auto"
                    final_updates[k] = v
                    if v == "off" and "math_ocr_enabled" not in updates:
                        final_updates["math_ocr_enabled"] = False
                elif k == "question_ocr_engine":
                    v = str(v).strip().lower()
                    if v not in ("groq", "gemini", "tesseract", "off"):
                        v = "groq"
                    final_updates[k] = v
                    if v == "off" and "question_ocr_enabled" not in updates:
                        final_updates["question_ocr_enabled"] = False
                elif k == "auto_whole_pdf_ocr_threshold":
                    try:
                        final_updates[k] = max(0, int(v))
                    except Exception:
                        final_updates[k] = 3
                elif k == "auto_whole_pdf_ocr_engine":
                    v = str(v).strip().lower()
                    if v not in ("auto", "groq", "gemini", "tesseract", "off"):
                        v = "auto"
                    final_updates[k] = v
                elif k in ("groq_api_key", "google_api_key", "gemini_api_key"):
                    val_str = str(v).strip()
                    if val_str:
                        target_env = "GROQ_API_KEY" if k == "groq_api_key" else "GOOGLE_API_KEY"
                        os.environ[target_env] = val_str
                        final_key = "groq_api_key" if k == "groq_api_key" else "google_api_key"
                        final_updates[final_key] = val_str
                else:
                    final_updates[k] = str(v)
            
            # Now safely write to DB without duplicating
            for k, v in final_updates.items():
                val_json = json.dumps(v)
                setting = db.query(PipelineSetting).filter(PipelineSetting.key == k).first()
                if setting:
                    setting.value = val_json
                    setting.updated_at = datetime.utcnow()
                else:
                    setting = PipelineSetting(key=k, value=val_json)
                    db.add(setting)
            
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.error("[ConfigManager] Failed to update settings in DB: %s", e)
    
    # Reload fresh state
    return get_all_settings()
