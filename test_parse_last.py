import re
import json

raw_text = """
Question1:-
Match List - I with List - II and select the correct answer from the codes given below.
ലിസ്റ്റ് - I നെ ലിസ്റ്റ് - II മായി യോജിപ്പിച്ച് താഴെ കൊടുത്തിരിക്കുന്ന കോഡുകളിൽ നിന്ന് ശരിയായ ഉത്തരം തിരഞ്ഞെടുക്കുക.
List - I List - II
ലിസ്റ്റ് - I ലിസ്റ്റ് - II

A. Mooshakavamsa Kavya i. Hendrik Van Rheede
മൂഷകവംശ കാവ്യം ഹെൻഡ്രിക് വാൻ റീഡ്
B. Hortus Malabaricus ii. Luis de Camoes
ഹോർട്ടസ് മലബാറിക്കസ് ലൂയിസ് ഡി കാമോസ്
C. Tuhafat ul Mujahiddin iii. Athula
തുഹ്ഫത്തുൽ മുജാഹിദീൻ അതുല
D. The Lusiads iv. William Logan
ലൂസിയാഡുകൾ വില്യം ലോഗൻ
E. Malabar Manual v. Shaikh Zainuddin
മലബാർ മാനുവൽ ശൈഖ് സൈനുദ്ദീൻ

A:-A - ii, B - iv, C - v, D - i, E - iii
B:-A - iii, B - iv, C - ii, D - v, E - i
C:-A - iii, B - i, C - v, D - ii, E - iv
D:-A - iii, B - ii, C - v, D - i, E - iv

Correct Answer:- Option-C
"""

content = raw_text.split("Question1:-")[1].strip()

# Find all potential options
options_pattern = re.compile(r'(?:^|\s)(?:\()?([A-D])(?:\)|\.|:-)\s*', re.IGNORECASE | re.MULTILINE)
matches = list(options_pattern.finditer(content))

# To find the true options block, we expect A, B, C, D to appear in sequence near the end.
# A robust way is to just find the LAST occurrence of 'A', 'B', 'C', 'D' markers.
# Let's collect all matches and their positions.
if matches:
    # We want the last A, last B, last C, last D
    last_positions = {}
    for m in matches:
        letter = m.group(1).upper()
        last_positions[letter] = m
    
    # Are A, B, C, D all present in last_positions?
    if 'A' in last_positions and 'B' in last_positions:
        # Sort these last matches by their start index
        true_option_matches = sorted(last_positions.values(), key=lambda m: m.start())
        
        # The stem is everything before the true 'A' (which should be the first in true_option_matches)
        first_opt_match = true_option_matches[0]
        stem = content[:first_opt_match.start()].strip()
        
        options = {}
        for i, match in enumerate(true_option_matches):
            opt_letter = match.group(1).upper()
            start_idx = match.end()
            end_idx = true_option_matches[i+1].start() if i + 1 < len(true_option_matches) else len(content)
            
            opt_text = content[start_idx:end_idx].strip()
            # Remove Correct answer from the last option if present
            ans_match = re.search(r'(?:Ans(?:wer)?|Correct(?:\s*Answer)?)\s*[:\-]*\s*(?:is\s*)?(?:Option\s*[:\-]*\s*)?([A-D])\b', opt_text, re.IGNORECASE)
            if ans_match:
                opt_text = opt_text[:ans_match.start()].strip()
                
            options[opt_letter] = opt_text
            
        print("Stem:", stem)
        print("Options:", json.dumps(options, indent=2))
