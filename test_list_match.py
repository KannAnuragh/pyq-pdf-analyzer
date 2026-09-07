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

def extract_questions(text):
    extracted = []
    # Split text into blocks starting with Question/Q followed by number
    pattern = r"(?:Question|Q)\s*(\d+)[\s.:-]*\n"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    # parts[0] is everything before first question
    for i in range(1, len(parts), 2):
        q_num = int(parts[i])
        block = parts[i+1]
        
        # Look for Correct Answer
        marked_answer = None
        ans_match = re.search(r"Correct Answer[^\n:]*[:-]\s*Option[-\s]*([A-D])", block, re.IGNORECASE)
        if not ans_match:
            ans_match = re.search(r"Correct Answer[^\n:]*[:-]\s*([A-D])", block, re.IGNORECASE)
            
        if ans_match:
            marked_answer = ans_match.group(1).upper()
            
        # Parse Options A, B, C, D
        options = {}
        stem = block
        
        # Find all option markers
        opt_pattern = r"(?:\n|^)([A-D])[:-](.*?)(?=(?:\n[A-D][:-]|\nCorrect Answer|$))"
        opt_matches = list(re.finditer(opt_pattern, block, re.IGNORECASE | re.DOTALL))
        
        if opt_matches:
            for match in opt_matches:
                opt_letter = match.group(1).upper()
                opt_text = match.group(2).strip()
                options[opt_letter] = opt_text
            # Stem is everything before the first option
            stem = block[:opt_matches[0].start()].strip()
            
            # if Correct Answer is above options, stem might need to end there instead
            if ans_match and ans_match.start() < opt_matches[0].start():
                stem = stem[:ans_match.start()].strip()
        
        extracted.append({
            "question_number": q_num,
            "stem": stem,
            "options": options,
            "marked_answer": marked_answer
        })
    return extracted

res = extract_questions(raw_text)
print(json.dumps(res, indent=2, ensure_ascii=False))

