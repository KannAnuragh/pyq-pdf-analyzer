import re

text = "Hello  world\n\nThis is\ta test."
parts = re.split(r'(\s+)', text)

offset = 0
for part in parts:
    if part.isspace() or not part:
        print(f"WS: {repr(part)} at {offset}")
    else:
        print(f"WORD: {repr(part)} at {offset}")
    offset += len(part)

for m in re.finditer(r"[^\s]+", text):
    print(f"FINDITER: {repr(m.group(0))} at {m.start()}")
