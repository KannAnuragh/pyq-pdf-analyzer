import re

def wrap_bare_math_test(text):
    # Mask already delimited math
    masked = []
    def _mask(m):
        idx = len(masked)
        masked.append(m.group(0))
        return f"__MASK{idx}__"
    s = re.sub(r'\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\$[^\$]+?\$|\\\([\s\S]+?\\\)', _mask, text)

    # Wrap bare scientific notation like 26.4 \times 10^{-12} or 1.5 \times 10^{-16}
    def _wrap_sci(m):
        return f"${m.group(0)}$"

    # Match: (number \times )? 10^{power} or 10^power (handling possible whitespace or \n)
    s = re.sub(
        r'(?<!\$)\b(?:\d+(?:\.\d+)?\s*(?:\\times|×)\s*)?10\^\{?-?[0-9]+\}?(?!\$)',
        _wrap_sci,
        s
    )
    s = re.sub(r'\$[^\$]+?\$', _mask, s)

    # Clean bare \text{...} if outside math
    s = re.sub(r'\\text\{([^{}]+)\}', r'\1', s)

    for i in reversed(range(len(masked))):
        s = s.replace(f"__MASK{i}__", masked[i])
    return s

t1 = "Two parallel large thin metal sheets have equal surface densities 26.4 \\times \n10^{-12} C/m^2 of opposite signs."
t2 = "1.5 \\times 10^{-16} N/C"
t3 = "3 \\times 10^{-10} N/C"
t4 = "3 \\text{ N/C}"
t5 = "density of 10^{-7} C/m"

print("1:", repr(wrap_bare_math_test(t1)))
print("2:", repr(wrap_bare_math_test(t2)))
print("3:", repr(wrap_bare_math_test(t3)))
print("4:", repr(wrap_bare_math_test(t4)))
print("5:", repr(wrap_bare_math_test(t5)))
