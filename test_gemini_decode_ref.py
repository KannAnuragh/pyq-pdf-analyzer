import os
import sys

# Try standard import first, fall back to adding backend to path
try:
    from google import genai
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
    import google.generativeai as genai

# Load env vars
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), 'backend', '.env'))

api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    print("No Gemini API key found in backend/.env")
    exit(1)

text = "cl-ky-kw-L-Øn\\pw cq]w\\¬In. Bcy-∑m-cmWv temI-Ønse ]cn-ip-≤-hw-i-sa∂pw Ah-cmWv temIw `cn-t°-≠-sX∂pw P¿a≥Im¿ Bcy-∑m-cm-sW∂pw ln‰ve¿ A`n-{]m-b-s∏´p."

prompt = f"""
This is Malayalam text typed using a legacy ASCII font (like ML-TTKarthika or FML) where English letters correspond to Malayalam characters.
Please transliterate/decode it back into proper Unicode Malayalam script. Return ONLY the Malayalam Unicode text, no other words.

Text:
{text}
"""

print("Asking Gemini to decode the legacy font...")
try:
    # Try the new SDK
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        print("--- Result ---")
        print(response.text)
    except AttributeError:
        # Fallback to old SDK
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        print("--- Result ---")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
