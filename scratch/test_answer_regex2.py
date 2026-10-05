import re
pattern = re.compile(
    r"(?:(?:^|\n|\\n|>|\}|\]|\.\s+)\s*(?:\*\*\s*(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\s*\*\*|\*?\*?(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\*?\*?\s*(?:[\:\-]+\s*|(?=\n|\\n|$))))|"
    r"(?:(?:[ \t]+|\t)(?:\*\*\s*(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\s*\*\*|\*?\*?(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\*?\*?\s*[\:\-]+\s*))",
    re.IGNORECASE
)
text = "1888\n**Answer:** B"
m = pattern.search(text)
print("MATCHED TEXT:", m.group(0))
print("TAIL:", text[m.end():])
