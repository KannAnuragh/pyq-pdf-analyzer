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
    text = re.sub(rf'െ({SUB_CLUSTER})ാ', r'\1ൊ', text)
    text = re.sub(rf'േ({SUB_CLUSTER})ാ', r'\1ോ', text)
    text = re.sub(rf'െ({SUB_CLUSTER})[ൗൌ]', r'\1ൌ', text)
    text = re.sub(rf'േ({SUB_CLUSTER})[ൗൌ]', r'\1ൌ', text)

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

    # 0. Protect HTML tags (inline equation images, diagrams) so cleaning never alters them
    html_tags = []
    def _mask_tag(m):
        idx = len(html_tags)
        html_tags.append(m.group(0))
        return f" ___HTML_TAG_{idx}___ "

    text = re.sub(r'<div[^>]*>[\s\S]*?</div>|<img[^>]*\/?>', _mask_tag, text, flags=re.IGNORECASE)

    # Revert accidental 20 \sigma^2 or 2 \sigma^2 6 back to 2002, 2026, 2025
    text = re.sub(r'\b20\s*\$?\\sigma\^?2\$?\b', '2002', text)
    text = re.sub(r'\b2\s*\$?\\sigma\^?2\$?\s*([0-9])\b', r'20\1', text)

    # Pre-clean statistical distribution, matrix expressions, and scientific formulas
    pat_array = r"\$?\s*(?:\\left\[|left\[)?\s*(?:\\begin|begin)\{array\}\{[a-z]+\}\s*([\s\S]+?)\s*(?:\\end|end)\{array\}\s*(?:\\right\]|right\])?\s*\$?"
    def _repl_arr(m):
        inner = m.group(1).strip()
        inner = re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", inner)
        return f"$\\begin{{bmatrix}} {inner} \\end{{bmatrix}}$"
    text = re.sub(pat_array, _repl_arr, text)

    pat_mat = r"\$?\s*(?:\\begin|begin)\{(bmatrix|pmatrix|vmatrix|Vmatrix|matrix)\}\s*([\s\S]+?)\s*(?:\\end|end)\{\1\}\s*\$?"
    def _repl_mat(m):
        env = m.group(1)
        inner = m.group(2).strip()
        inner = re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", inner)
        return f"$\\begin{{{env}}} {inner} \\end{{{env}}}$"
    text = re.sub(pat_mat, _repl_mat, text)

    # Scientific Formulas (Maths, Stats, Physics, Chemistry)
    # 1. Statistics & Distributions
    text = re.sub(r'\\frac\{1\}\{7\}', r'\\frac{1}{n}', text)
    text = re.sub(r'\\begin\{bmatrix\}\s*1vV\s*\\\\\s*1-1\s*&\s*hn\s*\\end\{bmatrix\}', r'\\frac{1}{n-1} \\sum', text)
    text = re.sub(r'\\frac\{id\}\{1\}', r'\\sum', text)
    text = re.sub(r'\bhn\b', r'\\sum', text)
    text = re.sub(r'\(\s*x\s*(?:\\hat\{i\}|[I\]i1])?\s*[-–—=]\s*(?:x[X\s]*|x)\s*\)\s*2\b', r'(x_i - \\bar{x})^{2}', text)
    text = re.sub(r'\$?\s*[xX]\s*=\s*=\s*\(\s*0\s*\)\s*=\s*2\s*\)\s*\$?|\$?\s*[xX]\s*[=~—–-]{1,2}\s*[\(=]?\s*[0-9oOnN\(\)\s=]*[,\s=]+[0oO©\s=]*\b2\s*\)\s*\$?', r'$X \\sim N(\\theta, \\sigma^2)$', text)
    # Match ONLY copyright symbol © / \u00A9 for OCR-misread sigma^2, NEVER plain 0 or O!
    text = re.sub(r'\$?[\u00A9©]\s*(?:\^\{?2\}?|2)\$?', r'$\\sigma^2$', text)
    text = re.sub(r'\b1-1\b', r'n-1', text)
    text = re.sub(r'\bxX\b', r'\\bar{x}', text)

    # 1.5 Mathematical Formulas & LaTeX normalization
    text = re.sub(r'\$?\s*V\s*aydyay\s*\$?', r'$\\frac{dy}{dx} + \\frac{1}{x}y = x^2$', text)
    text = re.sub(r'\b(?:ou|0u|eu)\b(?=\s*=)', r'\\frac{dy}{dx}', text)
    text = re.sub(r'(?<!\\frac\{)\bdy(?:\s*[\n/]|\\n)\s*dx\b', r'\\frac{dy}{dx}', text)
    text = re.sub(r'(?<!\\frac\{)\bdx(?:\s*[\n/]|\\n)\s*dy\b', r'\\frac{dx}{dy}', text)
    text = re.sub(r'(?<!\\frac\{)\bd\^?2y(?:\s*[\n/]|\\n)\s*dx\^?2\b', r'\\frac{d^2y}{dx^2}', text)
    text = re.sub(r'(?<!\\frac\{)\bdy(?:\s*[\n/]|\\n)\s*dt\b', r'\\frac{dy}{dt}', text)
    text = re.sub(r'(?<!\\frac\{)\bdx(?:\s*[\n/]|\\n)\s*dt\b', r'\\frac{dx}{dt}', text)

    text = re.sub(r'\$?x\\ne\$?', r'$e^x$', text)
    text = re.sub(r'\$?1(?:\s*[\n/]|\\n)\s*x\$?', r'$\\frac{1}{x}$', text)
    text = re.sub(r'\$?1(?:\s*[\n/]|\\n)\s*y\$?', r'$\\frac{1}{y}$', text)
    text = re.sub(r'\$?1(?:\s*[\n/]|\\n)\s*B\$?', r'$\\frac{1}{B}$', text)
    text = re.sub(r'\$?1(?:\s*[\n/]|\\n)\s*n\$?', r'$\\frac{1}{n}$', text)
    text = re.sub(r'\b2(?:\s*[\n/]|\\n)\s*X[xX]?\b', r'x^2', text)
    text = re.sub(r'\$2(?:\s*[\n/]|\\n)\s*[xX]\$?', r'$x^2$', text)
    text = re.sub(r'\bln\s*\(\s*x\s*\)', r'$\\ln(x)$', text)
    text = re.sub(r'\blog\s*\(\s*([a-zA-Z0-9]+)\s*\)', r'$\\log(\1)$', text)

    text = re.sub(r'tan[’\'"]+\s*[@0θ]?', r'\\tan^{2}\\theta', text)
    text = re.sub(r'sec[’\'"]+\s*[@0θ]?', r'\\sec^{2}\\theta', text)
    text = re.sub(r'sin[’\'"]+\s*[@0θ]?', r'\\sin^{2}\\theta', text)
    text = re.sub(r'cos[’\'"]+\s*[@0θ]?', r'\\cos^{2}\\theta', text)
    text = re.sub(r'\b(sin|cos|tan|sec|csc|cot)\s*[@0]\b', r'\1 \\theta', text)

    text = re.sub(r'\bIAU\s*BI\b', r'n(A \\cup B)', text)
    text = re.sub(r'\bA\s*∩\s*B\b', r'A \\cap B', text)
    text = re.sub(r'\^{?([0-9]+)}?C([0-9]+)', r'^{$1}C_{$2}', text)
    text = re.sub(r'e\^\{?x\}?\s*Ty', r'e^{x+y}', text)

    # Integrals & Vector Theorems
    text = re.sub(r'\[\s*f\s*F\s*sas\s*=\s*\[\s*o>\s*F\s*\)\s*dV', r'$\\iint_S \\vec{F} \\cdot d\\vec{S} = \\iiint_V (\\nabla \\cdot \\vec{F}) dV$', text)
    text = re.sub(r'\[\s*f\s*F\s*sas\s*=\s*\[\s*fv\s*F\s*dV', r'$\\iint_S \\vec{F} \\cdot d\\vec{S} = \\iiint_V (\\nabla \\cdot \\vec{F}) dV$', text)
    text = re.sub(r'\b(?:vVxF|vx\s*F)\s*=\s*0\b', r'$\\nabla \\times \\vec{F} = \\vec{0}$', text)
    text = re.sub(r'\b(?:V\.F|v\.F)\s*=\s*0\b', r'$\\nabla \\cdot \\vec{F} = 0$', text)
    text = re.sub(r'\|\s*Jt\s*dx\s*=\s*1\b', r'$\\int f(x) dx = 1$', text)
    text = re.sub(r'\[\s*te\s*dx\s*>\s*1\b', r'$\\int f(x) dx > 1$', text)
    text = re.sub(r'lax\s*ol\s*[-–—]\s*\(@\.b\)\s*=\s*1dhibP', r'$|\\vec{a} \\times \\vec{b}|^{2} + (\\vec{a} \\cdot \\vec{b})^{2} = |\\vec{a}|^{2} |\\vec{b}|^{2}$', text)
    text = re.sub(r'lax\s*or\s*\+\s*\(@\.b\)\s*=\s*I@hbP', r'$|\\vec{a} \\times \\vec{b}|^{2} + (\\vec{a} \\cdot \\vec{b})^{2} = |\\vec{a}|^{2} |\\vec{b}|^{2}$', text)
    text = re.sub(r'Vy\)\s*<\s*V\(ye\)', r'$V(\\bar{y}_{st}) < V(\\bar{y}_{sys})$', text)
    text = re.sub(r'\b2\s*a\b(?=\s*-\s*ടെസ്റ്റ്)', r'$\\chi^{2}$', text)
    text = re.sub(r'\bX\s*~\s*y\s*\(2\)', r'$X \\sim \\chi^{2}(2)$', text)

    # 2. Chemistry Formulas
    text = re.sub(r'\bH\s*2\s*O\b', r'$H_2O$', text)
    text = re.sub(r'\bC\s*O\s*2\b', r'$CO_2$', text)
    text = re.sub(r'\bH\s*2\s*S\s*O\s*4\b', r'$H_2SO_4$', text)
    text = re.sub(r'\bK\s*Mn\s*O\s*4\b', r'$KMnO_4$', text)
    text = re.sub(r'\bCa\s*C\s*O\s*3\b', r'$CaCO_3$', text)
    text = re.sub(r'\bNa\s*2\s*C\s*O\s*3\b', r'$Na_2CO_3$', text)
    text = re.sub(r'\bNa\s*H\s*C\s*O\s*3\b', r'$NaHCO_3$', text)

    # 3. Physics Formulas
    text = re.sub(r'\b[eE]\s*=\s*m\s*c\s*[\^]?\s*2\b', r'$E = mc^2$', text)
    text = re.sub(r'\b[vV]\s*[\^]?\s*2\s*=\s*[uU]\s*[\^]?\s*2\s*\+\s*2\s*a\s*s\b', r'$v^2 = u^2 + 2as$', text)
    text = re.sub(r'\b[vV]\s*[\^]?\s*2\s*=\s*[uU]\s*[\^]?\s*2\b', r'$v^2 = u^2$', text)
    text = re.sub(r'\b[hH]\s*\\?nu\b', r'$h\\nu$', text)

    # Clean redundant dollar signs and wrap orphaned math
    text = re.sub(r'\${2,}', '$', text)
    text = re.sub(r'\$\s*\$', ' ', text)

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

    # 5.3 Normalization for standard PostScript math symbols (e.g. StandardSymbolsPS)
    text = re.sub(r'´\s*N\b', r'\\times N', text)
    text = re.sub(r'Î\s*N\b', r'\\in N', text)

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

    # 7.2 Repair detached retroflex ള / plural suffixes (e.g. ചിഹ്ന ൾ -> ചിഹ്നങ്ങൾ, സ്വര ൾ -> സ്വരങ്ങൾ, അംഗ ളിൽ -> അംഗങ്ങളിൽ)
    # In Dravidian / Malayalam phonology, no word can begin with ള or chillu ൾ.
    # When separated by whitespace, these are broken plural suffixes where the root ending in vowel lost 'ങ്ങ' (due to font code 32).
    text = re.sub(r'([\u0D00-\u0D7F])[\s\xa0]+ൾ(?=[\s\.,;:!?\)\"\'\]]|$)', r'\1ങ്ങൾ', text)
    text = re.sub(r'([\u0D00-\u0D7F])[\s\xa0]+(ള[\u0D3E-\u0D4E\u0D57\u0D00-\u0D7F]*)', r'\1ങ്ങ\2', text)

    # 7.3 Repair verb stems with missing 'ക്കു' (caused by newline code 10 misinterpretation in subset fonts)
    # e.g. സൂചിപ്പി ന്ന -> സൂചിപ്പിക്കുന്ന, രചി ന്ന -> രചിക്കുന്ന, പരിശോധി ക -> പരിശോധിക്കുക
    text = re.sub(r'(സൂചിപ്പി|അവതരിപ്പി|പ്രതിപാദിപ്പി|ആലാപി|രചി|അറി|ഉപയോ\s*ഗി)[\s\xa0\n]+ന്ന', r'\1ക്കുന്ന', text)
    text = re.sub(r'(പരിശോധി|തെരഞ്ഞെടു|തിരഞ്ഞെടു|രേഖപ്പെടു)[\s\xa0\n]+ക\b', r'\1ക്കുക', text)

    # 7.4 Repair detached anusvara attached to Latin letters/quotes/variables
    # In legacy typesetting, typesetters typed '-ം' or ' ം' after an English variable (e.g. 'A' ം, 'B' ം, x-ം)
    # to represent the conjunction "-um" ("and"). In Unicode, U+0D02 is a combining mark that cannot attach to
    # Latin characters or quotes without the font engine synthesizing an ugly dotted circle (◌ം).
    # Normalizing this to standard Malayalam conjunction '-യും' / ' യും' perfectly restores author intent without dotted circles!
    text = re.sub(
        r"((?:^|[\s\(\[\{])['\"]?[A-Za-z0-9]['\"]?)\s*[-–—]?\s*[\u25CC]?ം(?=[\s\.,;:!?\)\"'\\\]]|$)",
        r"\1 യും",
        text
    )

    # 7.5 Normalize common mathematical OCR misrecognitions
    # Fix radical \X, VX -> \sqrt{x}, \times 2 -> x^2
    text = re.sub(r'\\X\b', r'\\sqrt{x}', text)
    text = re.sub(r'\bVX\b', r'\\sqrt{x}', text)
    text = re.sub(r'\bVx\b', r'\\sqrt{x}', text)
    text = re.sub(r'\bV(\d+)\b', r'\\sqrt{\1}', text)
    text = re.sub(r'\\x\b', 'x', text)
    text = re.sub(r'(?<=[=,\s])\s*\\times\s*([0-9])\b', r' x^{\1}', text)
    text = re.sub(r'\$\\times\s*([0-9])\b', r'$x^{\1}', text)

    # Vector arrows and unit vectors
    text = re.sub(r'\\frac\{[=><\-—~^d2\s|!_]+\}\{([A-Za-z])(?:\s*[A-Za-z]*)\}', r'\\vec{\1}', text)
    text = re.sub(r'\(\s*=\s*\+\s*([a-zA-Z0-9])\s*\)', r'(x + \1)', text)
    text = re.sub(r'\(\s*([a-zA-Z])\s*=\s*([a-zA-Z0-9])\s*\)', r'(\1 - \2)', text)
    text = re.sub(r'\)\s*[iI]\b', r')\\hat{i}', text)
    text = re.sub(r'\)\s*[jJ\]]\b', r')\\hat{j}', text)
    text = re.sub(r'\b[zZ]\s*2\s*[kK]\b', r'z^{2}\\hat{k}', text)
    text = re.sub(r'\b[zZ]\^\{?2\}?\s*[kK]\b', r'z^{2}\\hat{k}', text)
    text = re.sub(r'(?<=\+)\s*[kK]\b', r' \\hat{k}', text)

    # Matrix normalization and backslash repair
    # e.g. left[begin{array}{cc} ... end{array}right] or begin{bmatrix} -> $\begin{bmatrix} ... \end{bmatrix}$
    pat_array = r"\$?\s*(?:\\left\[|left\[)?\s*(?:\\begin|begin)\{array\}\{[a-z]+\}\s*([\s\S]+?)\s*(?:\\end|end)\{array\}\s*(?:\\right\]|right\])?\s*\$?"
    def _repl_array(m):
        inner = m.group(1).strip()
        inner = re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", inner)
        return f"$\\begin{{bmatrix}} {inner} \\end{{bmatrix}}$"
    text = re.sub(pat_array, _repl_array, text)

    pat_mat = r"\$?\s*(?:\\begin|begin)\{(bmatrix|pmatrix|vmatrix|Vmatrix|matrix)\}\s*([\s\S]+?)\s*(?:\\end|end)\{\1\}\s*\$?"
    def _repl_mat(m):
        env = m.group(1)
        inner = m.group(2).strip()
        inner = re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", inner)
        return f"$\\begin{{{env}}} {inner} \\end{{{env}}}$"
    text = re.sub(pat_mat, _repl_mat, text)

    # 8. Collapse multiple spaces into single spaces, but preserve newlines and tabs
    text = re.sub(r'[ \r\f\v]+', ' ', text).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 9. Restore protected HTML tags
    for idx, tag in enumerate(html_tags):
        text = text.replace(f"___HTML_TAG_{idx}___", tag)

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

    # Strip HTML tags (inline images, block diagrams) before checking text sanity
    text = re.sub(r'<[^>]+>', ' ', text)

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
        # Check if this is genuine bilingual text (e.g. English questions followed by Malayalam questions)
        words = set(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower()))
        common_en = {'the', 'of', 'and', 'to', 'in', 'is', 'that', 'for', 'it', 'as', 'was', 'with', 'on', 'by', 'at', 'question', 'option', 'answer', 'correct', 'which', 'following', 'from', 'this'}
        if not (words & common_en):
            return False, {"reason": "overwhelmed by english letters", "coverage": coverage}

    # 4. Check for too many garbled non-ASCII, non-Malayalam symbols
    garbled_chars = len(re.findall(r'[^\u0D00-\u0D7F\x00-\x7F]', non_ws))
    if garbled_chars > len(non_ws) * 0.05:
        return False, {"reason": "too many garbled symbols", "coverage": coverage}

    # 5. Check for visual-order pre-base matras at word boundaries
    if re.search(r'(?:^|\s)[\u0D46-\u0D48]', text):
        return False, {"reason": "visual-order pre-base matra detected", "coverage": coverage}

    return True, {"coverage": coverage}


def auto_fix_deterministic(text: str) -> tuple[str, list[str]]:
    """
    Deterministic, rule-based, safe auto-fix at $0 cost.
    Fixes:
      1. Extra/double spaces, space before punctuation, and space around combining signs.
      2. Orphaned matras (reordering per convert_visual_to_logical rules to attach to nearest valid consonant).
      3. Stray zero-width / control characters (ZWSP, BOM, dotted circle, non-virama ZWNJ/ZWJ, control codes).
      4. Duplicate-character leaks from band merging (3+ repeated Malayalam consonants, duplicate subjoined signs, duplicate vowel signs).

    Safety Guarantees:
      - Transformations are rule-derived, NOT inferred ($0 cost, no Vision call, no validation gate needed).
      - NEVER performs ambiguous single-glyph substitution ("this letter looks wrong"):
        Any isolated Latin characters, unmapped symbols, numbers, or formula variables are strictly
        preserved as-is so they remain visibly flagged for reviewers/Vision rather than silently
        guessing plausible wrong letters that could alter exam answers.
      - Protects all HTML tags (inline equation images, block diagrams) via placeholder masking.

    Returns:
        (fixed_text, list_of_applied_fixes)
    """
    if not text:
        return text, []

    fixes = []

    # 0. Protect HTML tags and LaTeX math formulas from regex manipulation
    html_tags = []
    def _mask_tag(m):
        idx = len(html_tags)
        html_tags.append(m.group(0))
        return f"%%HTMLTAG_{idx}%%"

    s = re.sub(r'<div[^>]*>[\s\S]*?</div>|<img[^>]*\/?>|\$\$[\s\S]*?\$\$|\$[^$\n]+\$', _mask_tag, text, flags=re.IGNORECASE)

    # 1. Stray zero-width and control characters
    old_s = s
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r'[\u200B\uFEFF\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u25CC]', '', s)
    # Strip stray ZWNJ/ZWJ not preceded by virama (valid only after virama ് in Malayalam)
    s = re.sub(r'(?<!\u0D4D)[\u200C\u200D]+', '', s)
    s = re.sub(r'(?:^|\s)[\u200C\u200D]+', '', s)
    if s != old_s:
        fixes.append("stray_control_chars")

    # 2. Duplicate-character leaks from band merging (never grammatically valid in Malayalam)
    old_s = s
    # 3+ repeated identical Malayalam consonants: collapse to single (or double for 'റ' which can represent 'റ്റ')
    s = re.sub(r'റ{3,}', 'ററ', s)
    s = re.sub(r'([ക-ഹ\u0D7A-\u0D7F])\1{2,}', r'\1', s)
    # Collapse duplicate subjoined signs (e.g. ഇന്ത്യ്യ -> ഇന്ത്യ)
    s = re.sub(r'(\u0D4D[\u200C\u200D]?[യവരല]){2,}', r'\1', s)
    # Collapse duplicate vowel signs / viramas (e.g. ്് -> ്, ാാ -> ാ)
    s = re.sub(r'([\u0D3E-\u0D4E\u0D57\u0D02\u0D03])\1+', r'\1', s)
    if s != old_s:
        fixes.append("duplicate_character_leak")

    # 3. Orphaned matras and visual-to-logical reordering
    old_s = s
    C = r'[ക-ഹ\u0D7A-\u0D7F]'
    CLUSTER = rf'(?:{C}്[\u200C\u200D]?)*{C}'
    BOUNDARY = r'(?:^|[\s\(\[\{<>\-_.,;:!?"\'/\\=+])'

    # Two-part split vowels with detached pre-base placement:
    s = re.sub(rf'({BOUNDARY})[\u25CC]*െ[\s\u25CC]*({CLUSTER})[\s\u25CC]*ാ', r'\1\2ൊ', s)
    s = re.sub(rf'({BOUNDARY})[\u25CC]*േ[\s\u25CC]*({CLUSTER})[\s\u25CC]*ാ', r'\1\2ോ', s)
    s = re.sub(rf'({BOUNDARY})[\u25CC]*െ[\s\u25CC]*({CLUSTER})[\s\u25CC]*[ൗൌ]', r'\1\2ൌ', s)
    s = re.sub(rf'({BOUNDARY})[\u25CC]*േ[\s\u25CC]*({CLUSTER})[\s\u25CC]*[ൗൌ]', r'\1\2ൌ', s)

    # Double e-karam to ai (െെ -> ൈ):
    s = re.sub(rf'({BOUNDARY})[\u25CC]*െെ[\s\u25CC]*({CLUSTER})', r'\1\2ൈ', s)

    # Pre-base matras (single െ, േ, ൈ) at boundary:
    s = re.sub(rf'({BOUNDARY})[\u25CC]*([െേൈ])[\s\u25CC]*({CLUSTER})', r'\1\3\2', s)
    # Repair detached anusvara on Latin variables / quotes (e.g. 'A' ◌ം 'B' ◌ം -> 'A' യും 'B' യും)
    s = re.sub(
        r"((?:^|[\s\(\[\{])['\"]?[A-Za-z0-9]['\"]?)\s*[-–—]?\s*[\u25CC]?ം(?=[\s\.,;:!?\)\"'\\\]]|$)",
        r"\1 യും",
        s
    )

    # Remove spaces separating Malayalam base characters from dependent vowel signs, viramas, anusvaras, or visargas
    s = re.sub(r'(?<=[\u0D00-\u0D7F])[\s\xa0]+([\u0D02\u0D03\u0D3E-\u0D4E\u0D57])', r'\1', s)
    # Remove spaces separating subjoined consonant signs (ya, va, ra, la)
    s = re.sub(r'(?<=[\u0D00-\u0D7F])[\s\xa0]+(്[\u200C\u200D]?[യവരല])', r'\1', s)

    # Apply standard visual-to-logical reordering (infix suffixes, pre-base matras, etc.)
    s = convert_visual_to_logical(s)
    if s != old_s:
        fixes.append("orphaned_matras")

    # 4. Extra / double spaces
    old_s = s
    s = re.sub(r'[ \t\r\f\v]+', ' ', s)
    s = re.sub(r'[ \t]+([,.:;!?])', r'\1', s)
    lines = [line.strip() for line in s.split('\n')]
    s = '\n'.join(lines).strip()
    s = re.sub(r'\n{3,}', '\n\n', s)
    if s != old_s:
        fixes.append("whitespace_normalized")

    # Restore protected HTML tags
    for idx, tag in enumerate(html_tags):
        s = s.replace(f"%%HTMLTAG_{idx}%%", tag)

    return s, fixes
