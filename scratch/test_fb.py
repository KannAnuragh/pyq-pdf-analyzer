import re

content = """A steady current of 8 mA flows through a wire. The number of electrons passing through a cross-section of the wire in 10 s is: [1 mark] (a) 4.0 \\times 10^{16} (b) 5.0 \\times 10^{17} (c) 1.6 \\times 10^{16} (d) 1.0 \\times 10^{17}"""

inline_pat = re.compile(r'(?:^|\s)[\(\[]?\s*([A-Ha-h1-8])\s*[\)\]\.\:\-]+\s')
fb_matches = []
for m in inline_pat.finditer(content):
    fb_matches.append((m.group(1), m))

if fb_matches:
    fb_best = []
    for i, m_info in enumerate(fb_matches):
        letter = m_info[0].upper()
        if letter != 'A' and letter != '1':
            continue
        curr_seq = [m_info]
        curr_letter = letter
        for next_m in fb_matches[i+1:]:
            next_letter = next_m[0].upper()
            if next_letter > curr_letter:
                letter_gap = ord(next_letter) - ord(curr_letter)
                if letter_gap <= 2:
                    curr_seq.append(next_m)
                    curr_letter = next_letter
        if len(curr_seq) > len(fb_best):
            fb_best = curr_seq
    if len(fb_best) >= 2:
        print("FOUND", len(fb_best))
        options = {}
        stem = content[:fb_best[0][1].start()].strip()
        for i, m_info in enumerate(fb_best):
            opt_letter = m_info[0]
            m = m_info[1]
            start_idx = m.end()
            end_idx = fb_best[i+1][1].start() if i + 1 < len(fb_best) else len(content)
            opt_text = content[start_idx:end_idx].strip()
            print("OPT", opt_letter, opt_text)
