"""
SMC Rule-Based FML (ML-TTKarthika) to Malayalam Unicode Transliteration Engine.

Implements the official Swathanthra Malayalam Computing (SMC) mapping rules
for converting legacy ASCII text (like ML-TTKarthika / FML) into proper Unicode Malayalam.
Handles pre-base matra reordering, conjunct ligatures, and chillu letters.
"""

import re
from typing import List, Tuple

# Official SMC ML-TTKarthika Ligature Mappings (Ordered from longest/most specific to shortest)
SMC_LIGATURES: List[Tuple[str, str]] = [
    ("sF", "ഐ"), ("sm", "ൊ"), ("tm", "ോ"), ("su", "ൌ"), ("ss", "ൈ"),
    ("Cu", "ഈ"), ("Du", "ഊ"), ("Hm", "ഓ"), ("Hu", "ഔ"),
    ("¡", "ക്ക"), ("¢", "ക്ല"), ("£", "ക്ഷ"), ("€", "ഗ്ഗ"), ("¥", "ഗ്ല"),
    ("Š", "ങ്ക"), ("§", "ങ്ങ"), ("š", "ച്ച"), ("©", "ഞ്ച"), ("ª", "ഞ്ഞ"),
    ("«", "ട്ട"), ("¬", "ണ്‍"), ("­", "ണ്ട"), ("ï", "ണ്ട"), ("®", "ണ്ണ"),
    ("¯", "ത്ത"), ("°", "ത്ഥ"), ("±", "ദ്ദ"), ("²", "ദ്ധ"), ("³", "ന്‍"),
    ("µ", "ന്ദ"), ("¶", "ന്ന"), ("·", "ന്മ"), ("¹", "പ്ല"), ("º", "ബ്ബ"),
    ("»", "ബ്ല"), ("Œ", "മ്പ"), ("œ", "മ്മ"), ("Ÿ", "മ്ല"), ("¿", "യ്യ"),
    ("À", "ര്‍"), ("Á", "റ്റ"), ("Â", "ൽ"), ("Ã", "ല്ല"), ("Ä", "ള്‍"),
    ("Å", "ള്ള"), ("Æ", "വ്വ"), ("Ç", "ശ്ല"), ("È", "ശ്ശ"), ("É", "സ്ല"),
    ("Ê", "സ്സ"), ("Ë", "ഹ്ല"), ("Ì", "സ്റ്റ"), ("Í", "ഡ്ഡ"), ("Î", "ക്ട"),
    ("Ï", "ബ്ധ"), ("Ð", "ബ്ദ"), ("Ñ", "ച്ഛ"), ("Ò", "ഹ്മ"), ("Ó", "ഹ്ന"),
    ("Ô", "ന്ധ"), ("Õ", "ത്സ"), ("Ö", "ജ്ജ"), ("×", "ണ്മ"), ("Ø", "സ്ഥ"),
    ("Ù", "ന്ഥ"), ("Ú", "ജ്ഞ"), ("Û", "ത്ഭ"), ("Ü", "ഗ്മ"), ("Ý", "ശ്ച"),
    ("Þ", "ണ്ഡ"), ("ß", "ത്മ"), ("à", "ക്ത"), ("á", "ഗ്ന"), ("â", "ന്റ"),
    ("ã", "ഷ്ട"), ("ä", "റ്റ"), ("å", "ന്"), ("´", "ന്ത"), ("¸", "പ്പ"),
    ("¨", "ച്ച"), ("¦", "ങ്ക"), ("¼", "മ്പ"), ("½", "മ്മ"), ("¾", "മ്ല"),
    ("¤", "ഗ്ഗ"), ("þ", "-"), ("∂", "ന്ന"), ("æ", "കു"), ("ê", "രു"),
    ("ç", "ക്കു"), ("‰", "റ്റ"), ("‖", "പ്പ"), ("ÿ", "സ്ഥ"),
    ("≥", "ന്റ"), ("ƒ", "ൾ"), ("∏", "പ്പ"), ("™", "ഞ്ഞ"), ("‚", "െ"),
    ("√", "ല്ല"), ("≠", "ണ്ട"), ("‡", "ക്ത"), ("∑", "ന്ന"), ("›", "ശ്ച")
]

# SMC Standard Character Mappings
SMC_SINGLE_CHARS = {
    'w': 'ം', 'x': 'ഃ', 'A': 'അ', 'B': 'ആ', 'C': 'ഇ', 'D': 'ഉ', 'E': 'ഋ',
    'F': 'എ', 'G': 'ഏ', 'H': 'ഒ', 'I': 'ക', 'J': 'ഖ', 'K': 'ഗ', 'L': 'ഘ',
    'M': 'ങ', 'N': 'ച', 'O': 'ഛ', 'P': 'ജ', 'Q': 'ഝ', 'R': 'ഞ', 'S': 'ട',
    'T': 'ഠ', 'U': 'ഡ', 'V': 'ഢ', 'W': 'ണ', 'X': 'ത', 'Y': 'ഥ', 'Z': 'ദ',
    '[': 'ധ', '\\': 'ന', ']': 'പ', '^': 'ഫ', '_': 'ബ', '`': 'ഭ', 'a': 'മ',
    'b': 'യ', 'c': 'ര', 'd': 'റ', 'e': 'ല', 'f': 'ള', 'g': 'ഴ', 'h': 'വ',
    'i': 'ശ', 'j': 'ഷ', 'k': 'സ', 'l': 'ഹ', 'm': 'ാ', 'n': 'ി', 'o': 'ീ',
    'p': 'ു', 'q': 'ൂ', 'r': 'ൃ', 's': 'െ', 't': 'േ', 'v': '്', 'u': 'ൗ',
    'y': '്യ', 'z': '്വ', '{': '്ര'
}

def convert_fml_to_unicode(text: str) -> str:
    """
    Convert legacy FML/ML-TT ASCII text to standard Unicode Malayalam using SMC rules.
    """
    if not text:
        return ""
        
    # Phase 1: Convert multi-character ligatures & complex conjuncts
    for ascii_seq, uni_seq in SMC_LIGATURES:
        text = text.replace(ascii_seq, uni_seq)
        
    # Phase 2: Convert single character mappings
    buf = []
    for char in text:
        buf.append(SMC_SINGLE_CHARS.get(char, char))
    text = "".join(buf)
    
    # Phase 3: Re-order pre-base vowel matras (െ, േ, ൈ) to post-base position
    text = re.sub(r'([െേൈ])([ക-ഹാ-ൌ്യ്ര്വ]+)', r'\2\1', text)
    
    return text

