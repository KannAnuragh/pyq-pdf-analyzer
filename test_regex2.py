import re

text = '''FINAL ANSWER KEY
Question Paper Code: 144/2026/OL
Question1:- ഇൻലാ ് വെസൽആക് ് 2021 കാര നിയമളെഎ ചാപ് കളായിതിരിിരി ?
        A:- X
        B:- XV
        C:- X X
        D:- XVIII
        Correct Answer:- Option-D
Question2:-
ഇൻലാ ് വെസൽആക് ് 2021 കാരംഉൾനാടൻജലാശയളിൽ 1.5 മീ ർ   തിരമാല )ടെഉയരം
ഉ   *ലളെഏത് സോണിൽഉൾെ ./0ിയിരി ?
        A:- സോൺ 1
        B:- സോൺ 2
        C:- സോൺ 3
        D:- ഇവയൊ മ2
        Correct Answer:- Option-A
'''

for opt_text in ["Correct Answer:- Option-D", "Ans: B", "Answer: C"]:
    ans_match = re.search(r'(?:Ans|Answer|Correct(?:\s*Answer)?)\s*[:\-]?\s*(?:Option\s*[\-:]?\s*)?([A-D])', opt_text, re.IGNORECASE)
    if ans_match:
        print(opt_text, "->", ans_match.group(1).upper())
