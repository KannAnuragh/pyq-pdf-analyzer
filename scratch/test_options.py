import re

def separate_stem_and_options_new(content: str):
    # Option pattern expanded to A-H
    opt_pattern = re.compile(r'(?:^|\n|\s+)(?:\()?([A-Ha-h])(?:\)|\.|:-)\s*')
    matches = []
    for m in opt_pattern.finditer(content):
        prefix = content[:m.start(1)].lower()
        if re.search(r'(?:assertion|reason)\s*\($', prefix):
            continue
        matches.append(m)
    
    best_sequence = []
    for i, m in enumerate(matches):
        curr_seq = [m]
        curr_letter = m.group(1).upper()
        for next_m in matches[i+1:]:
            next_letter = next_m.group(1).upper()
            
            # Enforce that the next letter is strictly greater
            if next_letter > curr_letter:
                # Calculate distance between this option and the previous one
                distance = next_m.start() - curr_seq[-1].end()
                
                # If they are absurdly far apart (e.g. > 1500 chars), it's probably a false positive
                if distance > 1500:
                    continue
                    
                # To prevent matching random A and then a random D later, we ensure the gap in letters is small
                letter_gap = ord(next_letter) - ord(curr_letter)
                if letter_gap <= 2: # Allow skipping at most 1 letter (e.g. A -> C if B is mangled)
                    curr_seq.append(next_m)
                    curr_letter = next_letter
        
        if len(curr_seq) > len(best_sequence):
            best_sequence = curr_seq

    if len(best_sequence) >= 2:
        options = {m.group(1).upper(): "" for m in best_sequence}
        return list(options.keys())
    return []

text_mcq = """
What is the color of the sky?
(A) Blue
(B) Green
(C) Red
(D) Yellow
"""

text_desc = """
(a) Explain gravity.
(b) Explain electromagnetism.
(c) Explain strong force.
(d) Explain weak force.
(e) Explain dark matter.
"""

text_false = """
In this paragraph we see (a) the first item.
Then a lot of text...
""" * 50 + "(b) the second item."

print("MCQ:", separate_stem_and_options_new(text_mcq))
print("Desc:", separate_stem_and_options_new(text_desc))
print("False:", separate_stem_and_options_new(text_false))
