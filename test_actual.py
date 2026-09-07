import re
import json

raw_text = """
Question1:-
Match List - I with List - II and select the correct answer from the codes given below.
ലിസ്റ്റ് - I നെ ലിസ്റ്റ് - II മായി യോജിപ്പിച്ച് താഴെ കൊടുത്തിരിക്കുന്ന കോഡുകളിൽ നിന്ന് ശരിയായ ഉത്തരം തിരഞ്ഞെടുക്കുക.
List - I List - II
ലിസ്റ്റ് - I ലിസ്റ്റ് - II

a. Mooshakavamsa Kavya i. Hendrik Van Rheede
മൂഷകവംശ കാവ്യം ഹെൻഡ്രിക് വാൻ റീഡ്
b. Hortus Malabaricus ii. Luis de Camoes
ഹോർട്ടസ് മലബാറിക്കസ് ലൂയിസ് ഡി കാമോസ്
c. Tuhafat ul Mujahiddin iii. Athula
തുഹ്ഫത്തുൽ മുജാഹിദീൻ അതുല
d. The Lusiads iv. William Logan
ലൂസിയാഡുകൾ വില്യം ലോഗൻ
e. Malabar Manual v. Shaikh Zainuddin
മലബാർ മാനുവൽ ശൈഖ് സൈനുദ്ദീൻ

A:-a - ii, b - iv, c - v, d - i, e - iii
B:-a - iii, b - iv, c - ii, d - v, e - i
C:-a - iii, b - i, c - v, d - ii, e - iv
D:-a - iii, b - ii, c - v, d - i, e - iv

Correct Answer:- Option-C
"""

q_prefix_pattern = r'(?:Question\s*|Q\s*)?\d+(?:[\.\)-]|:-)'
blocks = re.split(fr'\n(?={q_prefix_pattern})|^(?={q_prefix_pattern})', raw_text, flags=re.MULTILINE | re.IGNORECASE)

extracted = []
for block in blocks:
    block = block.strip()
    if not block: continue
    q_match = re.match(r'^(?:Question\s*|Q\s*)?(\d+)(?:[\.\)-]|:-)\s*(.*)', block, flags=re.DOTALL | re.IGNORECASE)
    if not q_match: continue
    
    q_num = int(q_match.group(1))
    content = q_match.group(2)
    
    options_pattern = re.compile(r'(?:^|\s)(?:\()?([A-D])(?:\)|\.|:-)\s*', re.IGNORECASE | re.MULTILINE)
    parts = options_pattern.split(content)
    
    stem = parts[0].strip()
    options = {}
    marked_answer = None
    
    for i in range(1, len(parts) - 1, 2):
        opt_letter = parts[i].upper()
        opt_text = parts[i+1].strip()
        ans_match = re.search(r'(?:Ans(?:wer)?|Correct(?:\s*Answer)?)\s*[:\-]*\s*(?:is\s*)?(?:Option\s*[:\-]*\s*)?([A-D])\b', opt_text, re.IGNORECASE)
        if ans_match:
            marked_answer = ans_match.group(1).upper()
            opt_text = opt_text[:ans_match.start()].strip()
        options[opt_letter] = opt_text
        
    if not marked_answer:
        ans_match = re.search(r'(?:Ans(?:wer)?|Correct(?:\s*Answer)?)\s*[:\-]*\s*(?:is\s*)?(?:Option\s*[:\-]*\s*)?([A-D])\b', stem, re.IGNORECASE)
        if ans_match:
            marked_answer = ans_match.group(1).upper()
            stem = stem[:ans_match.start()].strip()
            
    extracted.append({
        "question_number": q_num,
        "stem": stem,
        "options": options,
        "marked_answer": marked_answer
    })

print(json.dumps(extracted, indent=2))
