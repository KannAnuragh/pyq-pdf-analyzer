import re

def heal_equation_multiplication(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    # Fuse number/operator outside $math$ into $number \times math$
    text = re.sub(
        r'(\b\d+(?:\.\d+)?)\s*(?:[×\*]|(?:\s+x\s+)|\s+times\s+)\s*\$([^$]+)\$',
        lambda m: f'${m.group(1)} \\times {m.group(2).strip()}$',
        text
    )

    # Fuse $math$ with trailing multiplication into $math \times number$
    text = re.sub(
        r'\$([^$]+)\$\s*(?:[×\*]|(?:\s+x\s+)|\s+times\s+)\s*(\b\d+(?:\.\d+)?)\b',
        lambda m: f'${m.group(1).strip()} \\times {m.group(2)}$',
        text
    )

    # Format units directly following math expressions: e.g. "$5 \times 10^5$V/m" -> "$5 \times 10^5\text{ V/m}$"
    text = re.sub(
        r'\$([^$]+)\$\s*([a-zA-Z]+(?:\/[a-zA-Z]+)?)\b',
        lambda m: f'${m.group(1).strip()}\\text{{ {m.group(2)}}}$' if m.group(2).lower() not in ('is', 'and', 'or', 'to', 'in', 'if', 'the', 'then', 'when', 'where', 'as', 'at') else m.group(0),
        text
    )

    # Clean fraction units like \frac{10^5V}{m} -> \frac{10^5\text{ V}}{\text{m}}
    text = re.sub(
        r'(\\frac\{[^{}]*?10\^\{?\d+\}?)\s*([a-zA-Z]+)(\}\{[^{}]*?)([a-zA-Z]+)(\})',
        r'\1\\text{ \2}\3\\text{\4}\5',
        text
    )

    return text

print("Case 1:", heal_equation_multiplication(r'5 × $\frac{10^{5}V}{m}$'))
print("Case 2:", heal_equation_multiplication(r'$5 \times 10^5$V/m'))
print("Case 3:", heal_equation_multiplication(r'Answer:\n(b) 5 × $\frac{10^5 V}{m}$'))
print("Case 4:", heal_equation_multiplication(r'$8 \times 10^{6}$V/m'))
