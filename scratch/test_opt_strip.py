import re

text = """A steady current of 8 mA flows through a wire. The number of electrons passing through a cross-section of the wire in 10 s is: [1 mark]
(a) 4.0 \\times 10^{16}
(b) 5.0 \\times 10^{17}
(c) 1.6 \\times 10^{16}
(d) 1.0 \\times 10^{17}
"""

opts = {}
s = text.strip()

trailing_opt_pat = re.compile(
    r'(?:(?:\n|\s{2,}|\.\s+|[ \t]+)\(([A-Ha-h1-8])\)[\.\:\-]?\s*|(?:\n|\s{2,}|\.\s+|[ \t]+)([A-Ha-h1-8])[\.\)\:-]\s+)([^ \n\r\t][^\n\r]*)$',
    re.IGNORECASE
)

while True:
    m = trailing_opt_pat.search(s)
    if not m:
        break
    letter = (m.group(1) or m.group(2)).upper()
    val = m.group(3).strip()
    if s[m.start():].strip():
        s = s[:m.start()].strip()
        if letter not in opts or not opts[letter]:
            opts[letter] = val

print("Remaining stem:", repr(s))
print("Extracted opts:", opts)
