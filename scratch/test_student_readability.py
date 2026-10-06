import re

text = """The slope of the graph between the stopping potential ($V_0$) and the frequency ($\\nu$) is $\\frac{h}{e}$, whereh
$$isPlanck'sconstantande$$
is the charge of an electron."""

def heal_student_readability(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    # 1. Un-mash words and variables: whereh -> where $h$, ande -> and $e$
    s = re.sub(r"\b(where|when|let|if|with|given|here)([a-zA-Z])\b", lambda m: f"{m.group(1)} ${m.group(2)}$", text)
    s = re.sub(r"\b(and)([a-zA-RT-Z])\b", lambda m: f"{m.group(1)} ${m.group(2)}$", s)
    
    prose_words = {
        "is", "are", "was", "were", "the", "of", "and", "constant", "charge",
        "electron", "mass", "velocity", "potential", "frequency", "energy",
        "force", "field", "value", "between", "with", "from", "where", "planck"
    }
    def _fix_math_block(match):
        inner = match.group(1).strip()
        words_in_inner = [w.lower() for w in re.findall(r"[a-zA-Z]{2,}", inner)]
        matching = [w for w in words_in_inner if any(p in w for p in prose_words)]
        if len(matching) >= 2 or "planck" in inner.lower():
            if " " not in inner and len(inner) > 10:
                unmashed = inner
                unmashed = re.sub(r"^is(?=[A-Z])", "is ", unmashed)
                unmashed = re.sub(r"([a-z])([A-Z])", r"\1 \2", unmashed)
                unmashed = re.sub(r"([a-zA-Z]+'s)([a-zA-Z]+)", r"\1 \2", unmashed)
                unmashed = re.sub(r"([a-zA-Z]+)(and)([a-zA-Z]+)", r"\1 and \3", unmashed)
                unmashed = re.sub(r"\s{2,}", " ", unmashed).strip()
                # Wrap single-letter variable at end or standalone (e.g. "and e" -> "and $e$")
                unmashed = re.sub(r"(?<!['\w\\])\b([b-hj-zB-HJ-Z])\b", r"$\1$", unmashed)
                return unmashed
            else:
                cleaned = re.sub(r"(?<!['\w\\])\b([b-hj-zB-HJ-Z])\b", r"$\1$", inner)
                return cleaned
        return match.group(0)

    s = re.sub(r"\$\$([\s\S]+?)\$\$", _fix_math_block, s)
    s = re.sub(r"\$([^\$\r\n]+?)\$", _fix_math_block, s)
    return s

healed = heal_student_readability(text)
print("ORIGINAL:")
print(text)
print("\nHEALED:")
print(healed)
