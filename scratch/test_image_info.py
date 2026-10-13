import sqlite3
import json

db = sqlite3.connect("data/pdf_qa.db")
c = db.cursor()
c.execute("SELECT id, result_json FROM jobs ORDER BY created_at DESC LIMIT 1")
row = c.fetchone()
if not row:
    print("No jobs")
    exit(0)

job_id, result_json = row
result_obj = json.loads(result_json)

for q in result_obj.get("questions", []):
    print(f"Q {q.get('question_number')}: hint_page={q.get('page_number')}")
