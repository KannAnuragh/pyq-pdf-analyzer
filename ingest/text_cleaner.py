"""
Text cleaner for Malayalam PDF extractions.

Handles universal text normalization operations:
- Visual-to-logical matra reordering
- Unicode normalization (NFC)
- Control character removal
- Whitespace collapse
- Sanity checking for garbled text detection
"""

import re
import unicodedata


def convert_visual_to_logical(text: str) -> str:
    """
    Re-orders visual Malayalam sequences (e.g., e-karam + consonant) into logical Unicode sequences.
    Handles:
    - Pre-base vowel signs (െ, േ, ൈ)
    - Two-part split vowel signs:
        െ + consonant_cluster + ാ -> consonant_cluster + ൊ
        േ + consonant_cluster + ാ -> consonant_cluster + ോ
        െ + consonant_cluster + ൗ -> consonant_cluster + ൌ
        േ + consonant_cluster + ൗ -> consonant_cluster + ൌ
    - Partially swapped split vowel signs (consonant + െ + ാ -> consonant + ൊ)
    - Infix pre-base vowels inside words (-ന്റെ, -പ്പെ, -യെ, -ളെ)
    - Pre-base vowels preceded by word boundaries, independent vowels, vowel signs, virama, and chillus
    """
    if not text:
        return text

    # Strip dotted circle rendering artifacts early
    text = text.replace('\u25CC', '')

    # 0. Consolidate double e-karam to ai (െെ -> ൈ)
    text = re.sub(r'െെ', 'ൈ', text)

    C = r'[ക-ഹ\u0D7A-\u0D7F]'
    CLUSTER = rf'(?:{C}്[\u200C\u200D]?)*{C}'
    SUB_CLUSTER = rf'(?:്[\u200C\u200D]?[യവരല]|{CLUSTER})'
    VOWELS = r'[\u0D3E-\u0D4C\u0D57]'
    CHILLUS = r'[\u0D7A-\u0D7F]'
    NON_C = rf'(?:^|[^\u0D15-\u0D3A])'
    END_B = r'(?=$|[\s\(\[\{<>\-_.,;:!?"\'/\\=+])'

    # 1. Two-part split vowels with pre-base placement:
    text = re.sub(rf'({NON_C})െ({SUB_CLUSTER})ാ', r'\1\2ൊ', text)
    text = re.sub(rf'({NON_C})േ({SUB_CLUSTER})ാ', r'\1\2ോ', text)
    text = re.sub(rf'({NON_C})െ({SUB_CLUSTER})[ൗൌ]', r'\1\2ൌ', text)
    text = re.sub(rf'({NON_C})േ({SUB_CLUSTER})[ൗൌ]', r'\1\2ൌ', text)

    # 2. Partially swapped split vowels (e/ee after cluster followed by aa):
    text = re.sub(rf'({SUB_CLUSTER})ൊ', r'\1ൊ', text)
    text = re.sub(rf'({SUB_CLUSTER})ോ', r'\1ോ', text)
    text = re.sub(rf'({SUB_CLUSTER})െ[ൗൌ]', r'\1ൌ', text)

    # 3. Infix pre-base matras preceded by bare consonants in raw visual text:
    # (Must run BEFORE pre-base swap so words starting with െ (like െബന്റു, െസ്റ്റപ്പ്, െചയ്യണം)
    #  never have their e-karam confused with an infix matra!)

    # a. Genitive suffix -nte: െ + ന്റ -> ന്റ + െ (ന്റെ)
    text = re.sub(rf'({NON_C})െ(ന്റ)', r'\1\2െ', text)

    # b. Passive/auxiliary verb stem -ppe: െ + പ്പ -> പ്പ + െ (പ്പെ)
    text = re.sub(rf'({NON_C})െ(പ്പ)', r'\1\2െ', text)

    # c. Turning/carving/picking verb stem: തിര + െ + ഞ്ഞ -> തിര + ഞ്ഞ + െ (തിരഞ്ഞെടുക്കാൻ)
    text = re.sub(rf'({NON_C})െ(ഞ്ഞ)', r'\1\2െ', text)

    # d. Accusative/locative suffix -the: െ + ത്ത -> ത്ത + െ (ത്തെ)
    text = re.sub(rf'({NON_C})െ(ത്ത)', r'\1\2െ', text)

    # e. Intensive/reflexive suffix -nne: െ + ന്ന -> ന്ന + െ (ന്നെ / തന്നെ)
    text = re.sub(rf'({NON_C})െ(ന്ന)', r'\1\2െ', text)

    # f. Genitive suffix -de: െ + ട -> ട + െ (ടെ)
    # (Must NOT match when da already has another vowel sign like du in thiranjedukkuka or ariyappedunnu!)
    text = re.sub(rf'({NON_C})െ(ട)(?!{VOWELS})', r'\1\2െ', text)

    # g. Accusative suffix -ye / -yellaam: െ + യ -> യ + െ (യെ / യെല്ലാം)
    text = re.sub(rf'({NON_C})െ(യ)(?:ല്ലാം|െല്ലാം){END_B}', r'\1\2െല്ലാം', text)
    text = re.sub(rf'({NON_C})െ(യ(?:ൽ|ിൽ|േ)?){END_B}', r'\1\2െ', text)

    # h. Accusative plural suffix -le: െ + ള -> ള + െ (ളെ)
    text = re.sub(rf'({NON_C})െ(ള){END_B}', r'\1\2െ', text)

    # i. Pre-base ai-karam before consonant clusters:
    text = re.sub(rf'({NON_C})ൈ({CLUSTER})', r'\1\2ൈ', text)

    # j. Standalone visual typesetting words and known visual ligatures:
    text = re.sub(r'\bവളെര\b', 'വളരെ', text)
    text = re.sub(r'(?:^|(?<=[\s\(\[\{<>\-_.,;:!?"\'/\\=+]))വെര', 'വരെ', text)
    text = re.sub(r'പാർലെമന്റ്', 'പാർലമെന്റ്', text)
    text = re.sub(r'ടർപ്പൈന്റൻ', 'ടർപ്പന്റൈൻ', text)
    text = re.sub(r'കessori|കസേര|കേസര', 'കസേര', text)
    text = re.sub(r'ഓപ്പേറഷൻ', 'ഓപ്പറേഷൻ', text)
    text = re.sub(r'വട്ടേമശ', 'വട്ടമേശ', text)
    text = re.sub(r'സേമ്മളന', 'സമ്മേളന', text)
    text = re.sub(r'നിറേവറ്റ', 'നിറവേറ്റ', text)
    text = re.sub(r'ഭരണ്യേദശ|ഭരണപ്രേദശ', 'ഭരണപ്രദേശ', text)
    text = re.sub(r'മേഞ്ചരി', 'മഞ്ചേരി', text)
    text = re.sub(r'ആദ്യെത്ത', 'ആദ്യത്തെ', text)
    text = re.sub(r'വിപ്ലവെത്ത', 'വിപ്ലവത്തെ', text)
    text = re.sub(r'തീരെത്ത', 'തീരത്തെ', text)
    text = re.sub(r'\bതെന്ന\b', 'തന്നെ', text)
    text = re.sub(r'\bഇന്ത\b', 'ഇന്ത്യ', text)
    text = re.sub(r'\bഇന്തയിലെ\b', 'ഇന്ത്യയിലെ', text)
    text = re.sub(r'\bഇന്തയുടെ\b', 'ഇന്ത്യയുടെ', text)
    text = re.sub(r'കണ\.ാ', 'കണക്കാ', text)
    text = re.sub(r'കെണ്ടത്ത', 'കണ്ടെത്ത', text)
    text = re.sub(r'പ്രേവശന', 'പ്രവേശന', text)
    text = re.sub(r'താരതേമ്യ', 'താരതമ്യേ', text)
    text = re.sub(r'ഭിന്നേശഷി', 'ഭിന്നശേഷി', text)
    text = re.sub(r'ചുവെട', 'ചുവടെ', text)
    text = re.sub(r'സാ\s*ലൈ\s*്', 'സാറ്റലൈറ്റ്', text)
    text = re.sub(r'കാവ[\s\u25CC]*[്്യ]?[\s\u25CC]*ം', 'കാവ്യം', text)
    text = re.sub(r'കാവ[\s\u25CC]+്യ', 'കാവ്യ', text)
    text = re.sub(r'വില[\s\u25CC]*[്്യ]?[\s\u25CC]*ം', 'വില്യം', text)
    text = re.sub(r'സംഖ[\s\u25CC]*[്്യ]?[\s\u25CC]*(?=[കയ])', 'സംഖ്യ', text)

    # 4. Pre-base matras (single െ, േ, ൈ) preceded by non-consonant boundary or at word start:
    text = re.sub(rf'({NON_C})[\s\u25CC]*([െേൈ])[\s\u25CC]*({CLUSTER})', r'\1\3\2', text)

    # 5. Second pass for newly exposed pre-base matras:
    text = re.sub(rf'({NON_C})[\s\u25CC]*([െേൈ])[\s\u25CC]*({CLUSTER})', r'\1\3\2', text)

    # 6. Safety cleanup for impossible vowel collisions caused by visual typesetting
    text = re.sub(r'പ്പടെു', 'പ്പെടു', text)
    text = re.sub(r'ഞ്ഞടെു', 'ഞ്ഞെടു', text)
    text = re.sub(r'ടെു', 'ടു', text)
    text = re.sub(r'([െേൈ])([ുൂിീ])', r'\2', text)

    return text


def clean_indic_text(text: str, fonts: list[str] = None) -> str:
    """
    Universal cleaner for Indic text from PDF extractions.
    Handles: Unicode normalization, control char removal, visual-to-logical reorder, whitespace collapse.
    """
    if not text:
        return text

    # 1. Normalize Unicode to NFC format
    text = unicodedata.normalize("NFC", text)

    INDIC_RANGE = r'[\u0900-\u0DFF\u0600-\u08FF]'

    # 2. Remove line-break hyphens splitting Indic words
    text = re.sub(f'({INDIC_RANGE})[-\u2013\u2014\xad]\\s*({INDIC_RANGE})', r'\1\2', text)
    text = re.sub(f'({INDIC_RANGE})[-\u2013\u2014]({INDIC_RANGE})', r'\1\2', text)

    # 3. Strip invisible formatting artifacts, control characters, and dotted circle placeholders
    text = re.sub(r'[\u200B\uFEFF\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = text.replace('\u25CC', '')

    # 4. Resolve known residual subset font ligature artifacts wedged in Malayalam words
    # (Must run BEFORE convert_visual_to_logical so split vowels like േ + ദ + * + ാ -> ദ്യോ work properly!)
    wedged_map = {
        'F': '്യ',
        'W': '്വ',
        'B': 'ളു',
        'H': 'ക്ക',
        'e': '്യു',
        '3': 'ു',
        '5': '്ര',
        '*': '്യ',
        '<': '്വ',
        'X': 'ള്ള',
        'U': 'ന്ധ',
        'V': 'വ്വ',
        'Y': 'ട്ട',
        'C': 'ച്ച',
        '_': 'ന്ത',
    }
    for char, repl in wedged_map.items():
        text = re.sub(rf'(?<=[\u0D00-\u0D7F]){re.escape(char)}', repl, text)

    # 5. Convert visual-order native extractions to logical standard unicode
    text = convert_visual_to_logical(text)

    # 5.1 Universal fallback for common missing letters (like missing 'ya' sign or pre-base matra)
    text = re.sub(r'\bറഷ[\s\u25CC]*ൻ\b', 'റഷ്യൻ', text)
    text = re.sub(r'\bചനീസ്\b', 'ചൈനീസ്', text)
    text = re.sub(r'\bഫ്രഞ[\s\u25CC]*്\b', 'ഫ്രഞ്ച്', text)

    # 5.2 Universal comma remapping to 'സ്റ്റ'
    # Tolerates optional dotted circles (U+25CC) and spaces that are often wedged in by PDF parsers
    text = re.sub(r',[\s\u25CC]*([\u0D3E-\u0D4E\u0D57])', r'സ്റ്റ\1', text)
    # Catch any common English loan words that might not have a vowel sign
    text = re.sub(r'(?<=ക)[\s\u25CC]*,[\s\u25CC]*മർ', 'സ്റ്റമർ', text) # കസ്റ്റമർ (customer)
    text = re.sub(r'(?<=സി)[\s\u25CC]*,[\s\u25CC]*ം', 'സ്റ്റം', text) # സിസ്റ്റം (system)
    text = re.sub(r'(?<=മാ)[\s\u25CC]*,[\s\u25CC]*ർ', 'സ്റ്റർ', text) # മാസ്റ്റർ (master)

    # 6. Collapse duplicate subjoined signs (e.g. ഇന്ത്യ്യ -> ഇന്ത്യ)
    text = re.sub(r'(\u0D4D\u0D2F){2,}', r'\1', text)
    text = re.sub(r'(\u0D4D\u0D35){2,}', r'\1', text)
    # 7. Strip dotted circle artifacts (U+25CC) and repair detached combining signs/chillus/ligatures
    text = text.replace('\u25CC', '')
    # Remove spaces erroneously separating Malayalam base characters from dependent vowel signs, viramas, anusvaras, or visargas
    text = re.sub(r'(?<=[\u0D00-\u0D7F])[\s\xa0]+([\u0D02\u0D03\u0D3E-\u0D4E\u0D57])', r'\1', text)
    # Remove spaces separating subjoined consonant signs (e.g. ya-sign, va-sign, ra-sign, la-sign)
    text = re.sub(r'(?<=[\u0D00-\u0D7F])[\s\xa0]+(്[\u200C\u200D]?[യവരല])', r'\1', text)

    # 7.1 Universal repair of misdecoded space characters (e.g. TrueType subset fonts where space code 32 resolved to GID 32 glyph 'വ')
    # a. Malayalam character + 'വ' + Latin / Digits / Roman numerals (e.g. ഓപ്ഷൻവA -> ഓപ്ഷൻ A, അലക്സാണ്ടർവII -> അലക്സാണ്ടർ II, ഇംഗ്ലണ്ടിൽവ1688 -> ഇംഗ്ലണ്ടിൽ 1688)
    text = re.sub(r'([\u0D00-\u0D7F])വ([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9])വ([\u0D00-\u0D7F])', r'\1 \2', text)

    # b. Chillu + 'വ' followed by an Indic letter (e.g. ഈജിപ്ഷ്യൻവസംസ്ക്കാര -> ഈജിപ്ഷ്യൻ സംസ്ക്കാര, നേട്ടങ്ങളിൽവഉൾപ്പെടുന്നവ -> നേട്ടങ്ങളിൽ ഉൾപ്പെടുന്നവ)
    text = re.sub(r'([ൽൻർൺൾ])വ([അ-ഔക-ഹ])', r'\1 \2', text)

    # c. Chillu + 'വെ' + consonant (where pre-base 'െ' was merged onto misdecoded 'വ')
    # e.g. കണ്ടുപിടുത്തങ്ങളിൽവെപടാത്തത് -> കണ്ടുപിടുത്തങ്ങളിൽ പെടാത്തത്, ഫെഡറലിസത്തിൻവെവല്ലുവിളിയായിരിക്കുന്നത് -> ഫെഡറലിസത്തിൻ വെല്ലുവിളിയായിരിക്കുന്നത്
    text = re.sub(r'([ൽൻർൺൾ])വെപ', r'\1 പെ', text)
    text = re.sub(r'([ൽൻർൺൾ])വെവ', r'\1 വെ', text)

    # 7.2 Repair detached plural suffixes (e.g. ചിഹ്ന ൾ -> ചിഹ്നങ്ങൾ, സ്വര ൾ -> സ്വരങ്ങൾ, ഗമക ളൊന്നുമില്ലാതെ -> ഗമകങ്ങളൊന്നുമില്ലാതെ)
    # In Malayalam, chillu L 'ൾ' is never a standalone word. When separated by a space from a preceding Indic word,
    # it is a broken plural suffix '-ങ്ങൾ' (where 'ങ്ങ' was lost or misdecoded as space).
    text = re.sub(r'([\u0D00-\u0D7F])[\s\xa0]+ൾ(?=[\s\.,;:!?\)\"\'\]]|$)', r'\1ങ്ങൾ', text)
    text = re.sub(r'([\u0D00-\u0D7F])[\s\xa0]+ളൊ', r'\1ങ്ങളൊ', text)

    # 8. Collapse multiple spaces into single spaces, but preserve newlines
    text = re.sub(r'[ \t\r\f\v]+', ' ', text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def sanity_check_malayalam_text(text: str) -> tuple[bool, dict]:
    """
    Checks if extracted text is clean Malayalam Unicode or garbled.
    Returns (is_sane, metrics_dict).
    
    A text fails sanity if it contains ASCII characters or symbols
    wedged inside Malayalam character sequences — a telltale sign of 
    broken font CID mappings.
    """
    if not text.strip():
        return True, {"reason": "empty"}

    non_ws = re.sub(r'\s+', '', text)
    if not non_ws:
        return True, {"reason": "only whitespace"}

    # Count character types
    malayalam_chars = len(re.findall(r'[\u0D00-\u0D7F]', non_ws))

    # If there's no Malayalam at all, skip Malayalam-specific checks
    if malayalam_chars == 0:
        return True, {"reason": "no_malayalam", "coverage": 1.0}

    # 1. Check for ASCII/symbol characters wedged directly inside a Malayalam word (no whitespace)
    # This is the strongest signal of garbled font CID extraction.
    wedged_match = re.search(
        r'[\u0D00-\u0D7F]'                        # Malayalam char before
        r'[A-Za-z#$%&*+<>=@\\^_`|~{}\[\]!/]'      # ASCII artifact (not digits)
        r'[\u0D00-\u0D7F]',                       # Malayalam char after
        text
    )
    if wedged_match:
        return False, {
            "reason": "ascii artifact wedged inside malayalam word",
            "match": wedged_match.group(0),
            "coverage": malayalam_chars / len(non_ws)
        }

    # 2. Check for overall coverage — valid characters should be predominantly Malayalam + basic punctuation
    valid_chars = len(re.findall(r'[\u0D00-\u0D7F\dA-Za-z\.,\?!\-\(\)\[\]:;\"\'\u200d<>=/&_ ]', non_ws))
    coverage = valid_chars / len(non_ws)

    if coverage < 0.85:
        return False, {"reason": f"low valid coverage: {coverage:.2f}", "coverage": coverage}

    # 3. Check if English overwhelms Malayalam (sign of total CMap failure)
    english_chars = len(re.findall(r'[A-Za-z]', non_ws))
    if english_chars > malayalam_chars * 1.5 and malayalam_chars > 10:
        return False, {"reason": "overwhelmed by english letters", "coverage": coverage}

    # 4. Check for too many garbled non-ASCII, non-Malayalam symbols
    garbled_chars = len(re.findall(r'[^\u0D00-\u0D7F\x00-\x7F]', non_ws))
    if garbled_chars > len(non_ws) * 0.05:
        return False, {"reason": "too many garbled symbols", "coverage": coverage}

    # 5. Check for visual-order pre-base matras at word boundaries
    if re.search(r'(?:^|\s)[\u0D46-\u0D48]', text):
        return False, {"reason": "visual-order pre-base matra detected", "coverage": coverage}

    return True, {"coverage": coverage}
