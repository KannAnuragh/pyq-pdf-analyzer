import re
block = "Correct Answer:- Option-C"
print(re.search(r"Correct Answer[^\n:]*[:-]\s*Option[-\s]*([A-D])", block, re.IGNORECASE).groups())
