import re
_Q_HEADER_RE = re.compile(
    r"""^
    (?:
        (?:Question\s*(?:No\.?)?\s*|Q\.?\s*(?:No\.?)?\s*|Sl\.?\s*No\.?\s*|Problem\s*|Prob\.?\s*)?
        (\d{1,3})
        \s*
        (?:[.):\-]|--)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

cases = [
    "2.  Two concentric and coplanar",
    "3. Loop Q carries a current",
    "2 : 3. Loop Q"
]
for c in cases:
    m = _Q_HEADER_RE.match(c)
    if m:
        gap = len(c[m.end():]) - len(c[m.end():].lstrip())
        print(f"Match: {c!r} -> gap: {gap}, remainder: {c[m.end():].lstrip()!r}")
    else:
        print(f"No match: {c!r}")
