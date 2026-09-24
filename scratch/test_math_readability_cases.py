import re

KNOWN_MATH_COMMANDS = {
    'frac', 'dfrac', 'tfrac', 'sqrt', 'times', 'div', 'pm', 'mp', 'cdot', 'circ', 'degree',
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma',
    'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega',
    'neq', 'ne', 'approx', 'leq', 'geq', 'le', 'ge', 'infty', 'sum', 'prod', 'int', 'oint',
    'log', 'ln', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan',
    'in', 'notin', 'subset', 'subseteq', 'supset', 'supseteq', 'cup', 'cap', 'setminus',
    'partial', 'nabla', 'vec', 'hat', 'dot', 'ddot', 'bar', 'tilde', 'overline', 'underline',
    'begin', 'end', 'left', 'right', 'bmatrix', 'pmatrix', 'vmatrix', 'Vmatrix', 'Bmatrix', 'matrix', 'array',
    'sim', 'hbar', 'rightarrow', 'to', 'leftarrow', 'leftrightarrow', 'Rightarrow', 'Leftarrow', 'Leftrightarrow',
    'text', 'mathrm', 'mathbf', 'mathit', 'mathbb', 'mathcal', 'quad', 'qquad',
    'equiv', 'propto', 'perp', 'parallel', 'therefore', 'because', 'angle', 'triangle',
    'bullet', 'dots', 'cdots', 'ldots', 'over', 'choose', 'cases', 'hline'
}

COMMON_PROSE_WORDS = {
    "is", "are", "was", "were", "the", "of", "and", "constant", "charge",
    "electron", "mass", "velocity", "potential", "frequency", "energy",
    "force", "field", "value", "between", "with", "from", "where", "planck",
    "which", "that", "this", "given", "equal", "equals", "zero", "ratio",
    "graph", "slope", "stopping", "threshold", "work", "function", "kinetic",
    "state", "ground", "excited", "particle", "wavelength", "momentum"
}

def heal_student_readability(text: str) -> str:
    """
    Heals prose and un-mashes variables for student readability.
    1. Un-mashes mashed prepositions/conjunctions and variables:
       'whereh' -> 'where $h$', 'ande' -> 'and $e$', 'whenv' -> 'when $v$'
    2. Detects mistakenly math-wrapped natural language prose in $...$ or $$...$$
       and restores it to natural English prose with proper single-letter variable math delimiters.
    """
    if not text or not isinstance(text, str):
        return text

    # Step 1: Un-mash prepositions and single-letter variables in running text
    # e.g. whereh -> where $h$, letx -> let $x$, withv -> with $v$
    text = re.sub(
        r"\b(where|when|let|if|with|given|here)([a-zA-Z])\b",
        lambda m: f"{m.group(1)} ${m.group(2)}$",
        text,
        flags=re.IGNORECASE
    )
    # e.g. ande -> and $e$, andh -> and $h$ (exclude 'and a' or 'and I')
    text = re.sub(
        r"\b(and)([b-hj-zB-HJ-Z])\b",
        lambda m: f"{m.group(1)} ${m.group(2)}$",
        text,
        flags=re.IGNORECASE
    )

    # Step 2: Unwrap natural language prose mistakenly wrapped in $...$ or $$...$$
    def _fix_math_prose_block(match):
        raw_inner = match.group(1)
        inner = raw_inner.strip()
        if not inner:
            return match.group(0)

        # If it has standard math commands (\frac, \sqrt, \alpha, etc.), it's genuine math
        tex_cmds = set(re.findall(r"\\([a-zA-Z]+)", inner))
        if tex_cmds and any(cmd in KNOWN_MATH_COMMANDS for cmd in tex_cmds) and not bool(re.search(r'[\u0900-\u0DFF]', inner)):
            # Genuine math with commands and no bare Indic characters
            return match.group(0)

        # Check for prose words
        # Extract potential words
        words_in_inner = [w.lower() for w in re.findall(r"[a-zA-Z]{2,}", inner)]
        prose_matches = [w for w in words_in_inner if any(p in w for p in COMMON_PROSE_WORDS)]

        # If the block contains prose terms or contains an apostrophe like 's with words, or Indic chars without \text
        is_prose = False
        if bool(re.search(r'[\u0900-\u0DFF]', inner)) and '\\text' not in inner:
            is_prose = True
        elif len(prose_matches) >= 2 or "planck" in inner.lower() or "'s" in inner:
            is_prose = True
        elif len(prose_matches) == 1 and not any(op in inner for op in ["=", "+", "\\", "_", "^"]):
            # Single prose word without math operators
            is_prose = True

        if not is_prose:
            return match.group(0)

        # It is prose! Let's clean and un-mash it
        unmashed = inner
        # Separate mashed words if no spaces or very few spaces
        if " " not in unmashed or len(re.findall(r"\s+", unmashed)) < len(prose_matches):
            unmashed = re.sub(r"^is(?=[A-Z])", "is ", unmashed)
            unmashed = re.sub(r"([a-z])([A-Z])", r"\1 \2", unmashed)
            unmashed = re.sub(r"([a-zA-Z]+'s)([a-zA-Z]+)", r"\1 \2", unmashed)
            unmashed = re.sub(r"([a-zA-Z]+)(and)([a-zA-Z]+)", r"\1 and \3", unmashed)
            unmashed = re.sub(r"\b(constant)([a-zA-Z]+)", r"\1 \2", unmashed)
            unmashed = re.sub(r"\b(charge)([a-zA-Z]+)", r"\1 \2", unmashed)
            unmashed = re.sub(r"\s{2,}", " ", unmashed).strip()

        # Wrap isolated single-letter variables in $ (e.g. "and e" -> "and $e$")
        unmashed = re.sub(r"(?<!['\w\\])\b([b-hj-zB-HJ-Z])\b", r"$\1$", unmashed)

        # Ensure nice spacing
        return unmashed

    # Apply to display math $$...$$ and \[...\]
    text = re.sub(r"\$\$([\s\S]+?)\$\$", _fix_math_prose_block, text)
    text = re.sub(r"\\\[([\s\S]+?)\\\]", _fix_math_prose_block, text)

    # Apply to inline math $...$ and \(...\) strictly ensuring it's not part of $$
    text = re.sub(r"(?<!\$)\$(?!\$)([^\$\r\n]+?)(?<!\$)\$(?!\$)", _fix_math_prose_block, text)
    text = re.sub(r"\\\(([\s\S]+?)\\\)", _fix_math_prose_block, text)

    # Clean up mid-sentence line breaks created by unwrapped display blocks
    # e.g. "where $h$\nis Planck's" -> "where $h$ is Planck's"
    text = re.sub(r"(\$[a-zA-Z]\$|[a-zA-Z,])\s*\n\s*([a-z])", r"\1 \2", text)

    return text

# Test cases
cases = [
    (
        "The slope of the graph between the stopping potential ($V_0$) and the frequency ($\\nu$) is $\\frac{h}{e}$, whereh\n$$isPlanck'sconstantande$$\nis the charge of an electron.",
        "where $h$ is Planck's constant and $e$ is the charge of an electron."
    ),
    (
        "Here $$is Planck's constant and e$$ is electron charge.",
        "Here is Planck's constant and $e$ is electron charge."
    ),
    (
        "whereh $isPlanck'sconstantande$",
        "where $h$ is Planck's constant and $e$"
    ),
    (
        "Calculate $\\frac{h}{e}$ when $V_0 = 1.5\\text{ V}$ and frequency is $\\nu$.",
        "$\\frac{h}{e}$ when $V_0 = 1.5\\text{ V}$ and frequency is $\\nu$."
    ),
    (
        "$$E = mc^2$$ where mass is $m$ and velocity is $c$.",
        "$$E = mc^2$$ where mass is $m$ and velocity is $c$."
    ),
    (
        "പ്രകാശവൈദ്യുത സമവാക്യം: $h\\nu = h\\nu_0 + K_{max}$, whereh $$isPlanck'sconstantande$$ is the charge.",
        "where $h$ is Planck's constant and $e$ is the charge."
    ),
    (
        "[A] $5.5 \\times 10^{-19}\\text{ J}$",
        "[A] $5.5 \\times 10^{-19}\\text{ J}$"
    ),
    (
        "When $v = 3\\times 10^8\\text{ m/s}$, calculate $\\lambda$.",
        "When $v = 3\\times 10^8\\text{ m/s}$, calculate $\\lambda$."
    ),
    (
        "Letx be the position and letp be momentum.",
        "Let $x$ be the position and let $p$ be momentum."
    ),
    (
        "ഇവിടെ $പ്രകാശവൈദ്യുത സമവാക്യം$ കാണുക.",
        "ഇവിടെ പ്രകാശവൈദ്യുത സമവാക്യം കാണുക."
    )
]

for idx, (inp, expected_substr) in enumerate(cases, 1):
    res = heal_student_readability(inp)
    print(f"--- Case {idx} ---")
    print(f"Input:    {inp!r}")
    print(f"Output:   {res!r}")
    assert expected_substr in res, f"Expected {expected_substr!r} in {res!r}"

print("\nALL TEST CASES PASSED SUCCESSFULLY!")

