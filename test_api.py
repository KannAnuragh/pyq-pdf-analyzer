from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test():
    q_data = {
        "job_id": "test_job_123",
        "question_number": 8,
        "stem": "A small piece of paper is stuck on the surface of the glass sphere of radius 10 cm and refractive index 1.5. The paper is viewed from the diametrically opposite side.",
        "options": {},
        "marked_answer": None,
        "explanation": "(a)\ni a.\nNy\nപു\nu\n_\nv\n(b) Using,\nMm നു സുന്ന്യ\nv വ R\nHere, u=—20cm,R= -10, മവു = 1.5,n,=1\nUpon substituting and calculating\n1-1\nv.40\n$v = -40cm$"
    }
    
    response = client.post("/api/fix_question_with_ai", json=q_data)
    print("STATUS:", response.status_code)
    try:
        res = response.json()
        print("NEEDS GEMINI OCR:", res.get("needs_gemini_ocr"))
        print("EXPLANATION:", res.get("explanation"))
    except Exception as e:
        print("Error parsing response:", e)

if __name__ == "__main__":
    test()
