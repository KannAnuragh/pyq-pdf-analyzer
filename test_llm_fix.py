import os
from google import genai

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("NO API KEY")
    exit(1)

text = "Question1:- ഇൻലാ ് െവസൽആക് ് 2021 കാര നിയമെളഎ ചാപ് കളായിതിരിിരി ?"

prompt = f"""
This is extracted Malayalam text from a PDF. It has two issues:
1. Visual ordering: Some vowel signs are separated from consonants (e.g. 'െ' before 'വ' instead of 'വെ').
2. Missing ligatures: Some complex characters were dropped (like 'ങ്ങ', 'റ്റ').

Please reconstruct the proper Malayalam sentence. Return ONLY the reconstructed text, keeping numbers and English letters exactly as they are.

Text to fix:
{text}
"""

client = genai.Client(api_key=api_key)
response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
print(response.text)
