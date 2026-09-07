"""
Document Ingestion Pipeline.

Orchestrates extraction, chunking, embedding, and vector storage.
"""

import re
import time
import unicodedata
from typing import Tuple

from ingestion.extractor import extract_text_from_file
from ingestion.chunker import chunk_text
from providers.factory import embedding_model, get_vector_db_client


def fix_malayalam_pdf_font_artifacts(text: str) -> str:
    """
    Fixes pseudo-Unicode font artifacts from legacy PDF extractions (ML-TT / FML CMap glitches).
    Converts broken character combinations into standard, pristine Malayalam Unicode.
    """
    if not text:
        return text

    # 1. Ra-vattu reordering (്ര before consonant -> consonant + ്ര)
    text = re.sub(r'്ര([ക-ഹ])', r'\1്ര', text)
    text = re.sub(r'സ്്ര', 'സ്ത്ര', text)

    # 2. Specific Symbol mapping
    text = re.sub(r'Ω', 'മ്മ', text)
    text = re.sub(r'≤', 'ദ്ധ', text)
    text = re.sub(r'μ', 'ന്ദ', text)
    text = re.sub(r'⁄', 'ജ്ഞ', text)
    text = re.sub(r'≈', 'ള്ള', text)
    text = re.sub(r'˛', ' - ', text)
    text = re.sub(r'‘', 'ന്ധ', text) # ഗാ‘ിജി -> ഗാന്ധിജി

    # 3. Consonant Glitch Mappings
    text = re.sub(r'യ്യ', 'ർ', text) # വളയ്യ -> വളർ, അവയ്യ -> അവർ
    
    # "ന്ത" mapped to "ഗ്ല" in some contexts
    text = re.sub(r'ഇഗ്ല്യ', 'ഇന്ത്യ', text)
    text = re.sub(r'ചിഗ്ല', 'ചിന്ത', text)
    text = re.sub(r'സ്വാതഗ്ര്ല്യ', 'സ്വാതന്ത്ര്യ', text)
    text = re.sub(r'അഗ്ലർ', 'അന്തർ', text) # യ്യ became ർ already
    text = re.sub(r'അഗ്ലയ്യ', 'അന്തർ', text) 
    text = re.sub(r'അത്യഗ്ലാ', 'അത്യന്താ', text)
    
    # "ട്ട" mapped to "ന്ത" in some contexts
    text = re.sub(r'കുന്തി', 'കുട്ടി', text)
    text = re.sub(r'പന്തിക', 'പട്ടിക', text)
    text = re.sub(r'ബ്രിന്തീഷ്', 'ബ്രിട്ടീഷ്', text)
    text = re.sub(r'മുന്നാന്തുേ', 'മുന്നോട്ട്', text)
    text = re.sub(r'ന്തുേ', 'ട്ട്', text)

    # "ത്ത" mapped to "സ്ഥ" in suffixes
    text = re.sub(r'സ്ഥിെെ', 'ത്തിന്റെ', text)
    text = re.sub(r'സ്ഥിന്', 'ത്തിന്', text)
    text = re.sub(r'സ്ഥിലൂടെ', 'ത്തിലൂടെ', text)
    text = re.sub(r'സ്ഥെ', 'ത്തെ', text)
    text = re.sub(r'സ്ഥിണ്', 'ത്തിൽ', text)
    text = re.sub(r'സ്ഥരം', 'ത്തരം', text)
    text = re.sub(r'പ്പെടുസ്ഥ', 'പ്പെടുത്ത', text)
    text = re.sub(r'വളർസ്ഥ', 'വളർത്ത', text) # യ്യ already -> ർ
    text = re.sub(r'ചലുെസ്ഥി', 'ചെലുത്തി', text)
    text = re.sub(r'ഭാഗസ്ഥ്', 'ഭാഗത്ത്', text)
    text = re.sub(r'രംഗസ്ഥും', 'രംഗത്തും', text)
    text = re.sub(r'രംഗസ്ഥ്', 'രംഗത്ത്', text)
    text = re.sub(r'ദേശസ്ഥു', 'ദേശത്തു', text)

    # "ക്ക" mapped to "ത്ഥ"
    text = re.sub(r'ത്ഥുക', 'ക്കുക', text)
    text = re.sub(r'ത്ഥാൻ', 'ക്കാൻ', text)
    text = re.sub(r'ത്ഥാനാണ്', 'ക്കാനാണ്', text)
    text = re.sub(r'ത്ഥും', 'ക്കും', text)
    text = re.sub(r'ത്ഥുന്ന', 'ക്കുന്ന', text)
    text = re.sub(r'ത്ഥാറു', 'ക്കാറു', text)
    text = re.sub(r'ത്ഥാെണ്ടി', 'ക്കൊണ്ടി', text)
    text = re.sub(r'ത്ഥിണ്ണ്', 'ച്ചിട്ട്', text)
    text = re.sub(r'ത്ഥുറി', 'ക്കുറി', text)
    text = re.sub(r'ആത്ഥി', 'ആക്കി', text)
    text = re.sub(r'ഇരിത്ഥും', 'ഇരിക്കും', text)
    text = re.sub(r'ഇരിത്ഥണം', 'ഇരിക്കണം', text)
    text = re.sub(r'ഉണ്ടാത്ഥി', 'ഉണ്ടാക്കി', text)
    text = re.sub(r'ത്ഥ്', 'ക്ക്', text)
    text = re.sub(r'ത്ഥാരിണ്', 'ക്കാരിൽ', text)
    text = re.sub(r'മായ്യത്ഥറ്റ്', 'മാർക്കറ്റ്', text)

    # "ച്ച" mapped to "ണ്ണ"
    text = re.sub(r'ശഖേരിണ്ണ്', 'ശേഖരിച്ച്', text)
    text = re.sub(r'പഠിണ്ണ', 'പഠിച്ച', text)
    text = re.sub(r'ഉപയാേഗിണ്ണ്', 'ഉപയോഗിച്ച്', text)
    text = re.sub(r'ഉപയോഗിണ്ണ്', 'ഉപയോഗിച്ച്', text)
    text = re.sub(r'നിർദേശിണ്ണു', 'നിർദ്ദേശിച്ചു', text)
    text = re.sub(r'ശ്രമിണ്ണ', 'ശ്രമിച്ച', text)
    text = re.sub(r'ചിന്തിണ്ണ', 'ചിന്തിച്ച', text)
    text = re.sub(r'വളർണ്ണ', 'വളർച്ച', text)
    text = re.sub(r'ചർണ്ണ', 'ചർച്ച', text)
    text = re.sub(r'നിയമിണ്ണു', 'നിയമിച്ചു', text)
    text = re.sub(r'സിദ്ധർണ്ണ', 'സിദ്ധിച്ച', text) # സിദ്ധിച്ചു / സിദ്ധിച്ച
    text = re.sub(r'ഉൾണ്ണർ', 'ഉൾച്ചേർ', text)

    # 4. Suffix and Case marker corrections
    text = re.sub(r'ണ്ണ്', 'ച്ച്', text) 
    text = re.sub(r'ണ്(\s|[.,!?”]|$)', r'ൽ\1', text)
    text = re.sub(r'ന്റ(\s|[.,!?”]|$)', r'ൻ\1', text)

    # 5. Word Specific Corrections
    text = re.sub(r'കസ്േ\s*ല്ലാന്റഡയ്യഡ്േ|കസ്േ\s*ല്ലഡി|കസ്േ', 'കേസ് സ്റ്റഡി', text)
    text = re.sub(r'ക ത ല്ലാന്റഡർഡേ', 'സ്റ്റാൻഡേർഡ്', text) # ക ത ല്ലാന്റഡയ്യഡ്േ -> ക ത ല്ലാന്റഡർഡേ
    text = re.sub(r'ല്ലാന്റഡർഡേ', 'സ്റ്റാൻഡേർഡ്', text)
    text = re.sub(r'കസിേനെ', 'കേസിനെ', text)
    text = re.sub(r'എഗ്ല്\?', 'എന്ത്?', text)
    text = re.sub(r'എഗ്ലിന്\?', 'എന്തിന്?', text)
    text = re.sub(r'പവർപാേയിെ\s*്|പവർപാേയിെ', 'പവർപോയിന്റ്', text)
    text = re.sub(r'പ്രസെേഷനായി', 'പ്രസന്റേഷനായി', text)
    text = re.sub(r'അപൂർവവും|അപൂർവ', 'അപൂർവ്വ', text)
    text = re.sub(r'വറിേന്തതുമായ|വറിേന്തതുമായി', 'വേറിട്ടതുമായ', text)
    text = re.sub(r'വറിേ', 'വേറി', text)
    text = re.sub(r'സന്നീയ്യണ|സന്നീർണ', 'സങ്കീർണ്ണ', text)
    text = re.sub(r'മണ്േപ്പറഞ്ഞ', 'മേൽപ്പറഞ്ഞ', text)
    text = re.sub(r'സിദ്ധർണ്ണ|സി≤ിണ്ണ', 'സിദ്ധിച്ച', text)
    text = re.sub(r'ബാന്ന്', 'ബാങ്ക്', text)
    text = re.sub(r'രഖേപ്പെടു', 'രേഖപ്പെടു', text)
    text = re.sub(r'വാർത്താവിനിമയം', 'വാർത്താവിനിമയം', text)
    text = re.sub(r'ദനൈംദിന', 'ദൈനംദിന', text)
    text = re.sub(r'വളർന്നിരി', 'വളർന്നിരി', text)
    text = re.sub(r'ഒന്തെറേ', 'ഒട്ടേറെ', text)
    text = re.sub(r'സാബ്ബസ്ഥിക|സാബ്ബത്തിക', 'സാമ്പത്തിക', text)
    text = re.sub(r'കാച്ചഗ്രേിെെ', 'കോൺഗ്രസിന്റെ', text)
    text = re.sub(r'നാഷണൽ', 'നാഷണൽ', text)
    text = re.sub(r'വിദ്യായ്യഥിയായിരിക്കുബ്ബോൾ', 'വിദ്യാർത്ഥിയായിരിക്കുമ്പോൾ', text)
    text = re.sub(r'താഴിെലുകൾ', 'തൊഴിലുകൾ', text)
    text = re.sub(r'ഗുണകരമാകുമന്ന്െ', 'ഗുണകരമാകുമെന്ന്', text)
    text = re.sub(r'അദ്ദഹേം', 'അദ്ദേഹം', text)
    text = re.sub(r'ഇസ്ഥരത്തിൽ|ഇത്തരത്തിൽ', 'ഇത്തരത്തിൽ', text)
    text = re.sub(r'പ്രതിരോധിക്കാമന്ന്െ', 'പ്രതിരോധിക്കാമെന്ന്', text)
    text = re.sub(r'അഭിപ്രായപ്പെന്തു', 'അഭിപ്രായപ്പെട്ടു', text)
    text = re.sub(r'വാർധാ', 'വാർധാ', text)
    text = re.sub(r'ഉണ്പ്പാദന|ഉൽപ്പാദന', 'ഉൽപ്പാദന', text)
    text = re.sub(r'ഏതെന്നിലും', 'ഏതെങ്കിലും', text)
    text = re.sub(r'വണേം', 'വേണം', text)
    text = re.sub(r'നൽകണമന്നുെം', 'നൽകണമെന്നും', text)
    text = re.sub(r'വിഭാവനചയ്തെ', 'വിഭാവനചെയ്ത', text)
    text = re.sub(r'സാക്കിയ്യ', 'സാക്കിർ', text)
    text = re.sub(r'നതൃേത്വത്തിൽ', 'നേതൃത്വത്തിൽ', text)
    text = re.sub(r'ദശീയവിദ്യോഭ്യാസം|ദശീയേവിദ്യാഭ്യാസം', 'ദേശീയവിദ്യാഭ്യാസം', text)
    text = re.sub(r'തന്നിന്തു-ള്ള-ത്|തന്നിട്ടുള്ളത്', 'തന്നിട്ടുള്ളത്', text)
    text = re.sub(r'അഗ്ലർദേശീയത', 'അന്തർദേശീയത', text)
    text = re.sub(r'പാരബ്ബര്യ', 'പാരമ്പര്യ', text)
    text = re.sub(r'വേദൈശികമധോവിത്വം', 'വൈദേശികമേധാവിത്വം', text)
    text = re.sub(r'എതിർപ്പ്', 'എതിർപ്പ്', text)
    text = re.sub(r'മനിലാക്കിയത്', 'മനസ്സിലാക്കിയത്', text)
    text = re.sub(r'പസ്ഥാെബ്ബതാം', 'പത്തൊമ്പതാം', text)
    text = re.sub(r'ചർച്ചചെøുന്നത്', 'ചർച്ചചെയ്യുന്നത്', text)
    text = re.sub(r'ദശീൈയക്യേത്തിന്', 'ദേശീയൈക്യത്തിന്', text)
    text = re.sub(r'വഴിയാെരുക്കി', 'വഴിയൊരുക്കി', text)
    text = re.sub(r'അത്യന്താപേക്ഷിതമാണന്ന്െ', 'അത്യന്താപേക്ഷിതമാണെന്ന്', text)

    # Structural font artifact replacements
    text = re.sub(r'ത്മ([ളൾൽനമതകറ])', r'ങ്ങ\1', text)
    text = re.sub(r'പ്രയാേഗ', 'പ്രയോഗ', text)
    text = re.sub(r'പ്രതിഭാസ', 'പ്രതിഭാസ', text)
    text = re.sub(r'പ്രശ്ന', 'പ്രശ്ന', text)
    text = re.sub(r'പ്രക്രിയ', 'പ്രക്രിയ', text)
    text = re.sub(r'പ്രായാേഗിക', 'പ്രായോഗിക', text)
    text = re.sub(r'പ്രദേശ', 'പ്രദേശ', text)

    # SCERT Specific Ligature mappings
    text = re.sub(r'ഗ്ല', 'ന്ത', text)
    text = re.sub(r'കുന്തി', 'കുട്ടി', text)
    text = re.sub(r'സ്ഥ', 'ത്ത', text)
    text = re.sub(r'ത്ഥ', 'ക്ക', text)
    text = re.sub(r'ണ്ണ', 'ച്ച', text)
    
    # Specific FML Symbol Mappings observed in SCERT PDFs
    text = re.sub(r'∏', 'പ്പ', text)
    text = re.sub(r'≈', 'ള്ള', text)
    text = re.sub(r'≥', 'ന്റ', text)
    text = re.sub(r'≠', 'ണ്ട', text)
    text = re.sub(r'ƒ', 'ൾ', text)
    text = re.sub(r'™', 'ഞ്ഞ', text)
    text = re.sub(r'˛', ' - ', text)
    text = re.sub(r'•', 'ഗ്ല', text)
    text = re.sub(r'√', 'ല്ല', text)
    text = re.sub(r'∑', 'ന്ന', text)
    
    # Pre-base & post-base vowel matras re-ordering
    text = re.sub(r'ാേ', 'ോ', text)
    text = re.sub(r'ിെ', 'ി', text)
    text = re.sub(r'െൈ', 'ൈ', text)
    
    # Direct word patches for common extracted glitches
    text = re.sub(r'്രപിയെ|്രപിയെപ്പന്ത', 'പ്രിയപ്പെട്ട', text)
    text = re.sub(r'സമൂഹസ്ഥത്ഥുറിെണ്ണു', 'സമൂഹത്തെക്കുറിച്ച്', text)
    
    # Ra-vattu (്ര) Reordering: move ്ര from before consonant to after it
    # E.g. ്രപിയെ -> പ്രിയെ (if not caught above)
    text = re.sub(r'്ര([ക-ഹ])', r'\1്ര', text)

    # Clean up leftovers
    text = re.sub(r'-\s*യുള്ള', 'യുള്ള', text)
    text = re.sub(r'-\s*ജ്ഞരെ', 'ജ്ഞരെ', text)
    text = re.sub(r'\$', '', text) # Remove isolated $ bullet points from PDF
    
    # Specific fix for "ചെøുന്നത്" FML leftover
    text = re.sub(r'ചെøുന്നത്', 'ചെയ്യുന്നത്', text)
    text = re.sub(r'നയ്യക്കോഴ്ച', 'നേർക്കാഴ്ച', text)
    text = re.sub(r'സായ്യവലൗകിക', 'സാർവ്വലൗകിക', text)

    return text


def clean_indic_text(text: str) -> str:
    """
    Cleaner for Indic regional languages and PDF artifact glyphs.
    Covers: Malayalam, Devanagari, Tamil, Telugu, Kannada, etc.
    """
    # 0. Normalize Unicode to NFC format
    text = unicodedata.normalize("NFC", text)

    INDIC_RANGE = r'[\u0900-\u0DFF\u0600-\u08FF]'

    # 2. Remove line-break hyphens splitting Indic words (including soft hyphens and dashes)
    text = re.sub(f'({INDIC_RANGE})[-–—\xad]\\s*({INDIC_RANGE})', r'\1\2', text)
    # Clean mid-word hyphens (e.g. ദേശീയ-തയും -> ദേശീയതയും)
    text = re.sub(f'({INDIC_RANGE})[-–—]({INDIC_RANGE})', r'\1\2', text)

    # 3. Strip invisible formatting artifacts and control characters
    text = re.sub(r'[\u200B\uFEFF\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # 4. Post-process lingering FML legacy font artifacts if present
    if re.search(r'[≥‰ƒ∏™‚√≠‡∑›]', text):
        try:
            from ingestion.smc_fml_converter import convert_fml_to_unicode
            text = convert_fml_to_unicode(text)
        except Exception:
            pass

    # 5. Fix pseudo-Unicode font artifacts from legacy PDF extractions
    text = fix_malayalam_pdf_font_artifacts(text)

    # 6. Collapse multiple whitespaces/newlines into single spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def detect_indic_script(text: str) -> str:
    """
    Detects script type based on character frequency without ML models.
    Zero-dependency, runs in microseconds.
    """
    script_counts = {
        'devanagari': len(re.findall(r'[\u0900-\u097F]', text)),  # Hindi, Marathi, Sanskrit
        'bengali': len(re.findall(r'[\u0980-\u09FF]', text)),     # Bengali, Assamese
        'gurmukhi': len(re.findall(r'[\u0A00-\u0A7F]', text)),    # Punjabi
        'gujarati': len(re.findall(r'[\u0A80-\u0AFF]', text)),    # Gujarati
        'odia': len(re.findall(r'[\u0B00-\u0B7F]', text)),        # Odia
        'tamil': len(re.findall(r'[\u0B80-\u0BFF]', text)),       # Tamil
        'telugu': len(re.findall(r'[\u0C00-\u0C7F]', text)),      # Telugu
        'kannada': len(re.findall(r'[\u0C80-\u0CFF]', text)),     # Kannada
        'malayalam': len(re.findall(r'[\u0D00-\u0D7F]', text)),   # Malayalam
        'urdu': len(re.findall(r'[\u0600-\u06FF]', text)),        # Urdu
    }

    top_script, count = max(script_counts.items(), key=lambda x: x[1])
    return top_script[:10] if count > 5 else 'english'


def run_ingestion_pipeline(
    file_path: str, 
    filename: str, 
    user_id: str, 
    course_id: str, 
    document_id: str,
    is_global: bool = False
) -> Tuple[int, str]:
    """
    Run the full ingestion pipeline.
    
    Returns:
        (page_count, detected_language)
    """
    print(f"⏱️ [Pipeline] Starting ingestion for {filename}")
    
    # 1. Extract Text
    t0 = time.time()
    pages_text, page_count = extract_text_from_file(file_path)
    t1 = time.time()
    print(f"⏱️ [Pipeline] Extraction took {t1 - t0:.2f} seconds for {page_count} pages.")
    
    if not pages_text:
        raise ValueError("No text could be extracted from the PDF")
        
    # Clean text to remove line-break hyphens and OCR artifacts for all Indic scripts
    pages_text = [(page_num, clean_indic_text(text)) for page_num, text in pages_text]
        
    # 2. Detect Language (zero-dependency, character-range based)
    full_text_sample = " ".join([t for _, t in pages_text[:5]])
    language = detect_indic_script(full_text_sample)
    
    # 3. Chunk Text
    t2 = time.time()
    chunks = chunk_text(pages_text, filename)
    t3 = time.time()
    print(f"⏱️ [Pipeline] Chunking {len(pages_text)} pages took {t3 - t2:.2f} seconds, created {len(chunks)} chunks.")
    
    if not chunks:
        raise ValueError("Chunking resulted in 0 chunks")
        
    # 4. Generate Embeddings
    t4 = time.time()
    texts_to_embed = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts_to_embed)
    t5 = time.time()
    print(f"⏱️ [Pipeline] Embedding {len(chunks)} chunks took {t5 - t4:.2f} seconds.")
    
    # Filter out chunks that failed to embed (where embedding is None)
    valid_chunks = []
    valid_embeddings = []
    for c, emb in zip(chunks, embeddings):
        if emb is not None:
            valid_chunks.append(c)
            valid_embeddings.append(emb)
        else:
            print(f"⚠️ [Pipeline] Dropping chunk {c.get('page_number')} due to failed embedding on Cloudflare.", flush=True)
            
    # 5. Store in Vector DB
    t6 = time.time()
    if valid_chunks:
        vector_db = get_vector_db_client()
        vector_db.upsert_chunks(
            user_id=user_id,
            course_id=course_id,
            document_id=document_id,
            filename=filename,
            chunks=valid_chunks,
            embeddings=valid_embeddings,
            is_global=is_global
        )
    else:
        print(f"⚠️ [Pipeline] No valid chunks remaining to upsert for {filename}.", flush=True)
        
    t7 = time.time()
    print(f"⏱️ [Pipeline] Vector DB upsert took {t7 - t6:.2f} seconds.")
    
    # Calculate estimated token cost based on total embedded characters
    total_chars_embedded = sum(len(c["text"]) for c in valid_chunks) if valid_chunks else 0
    estimated_tokens = total_chars_embedded // 4
    print(f"💰 [Pipeline] Estimated token cost for '{filename}': ~{estimated_tokens} tokens", flush=True)
    
    print(f"⏱️ [Pipeline] Total ingestion time: {t7 - t0:.2f} seconds.")
    return page_count, language
