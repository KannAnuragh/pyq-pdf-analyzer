import sqlite3
import json

conn = sqlite3.connect("data/pdf_qa.db")
c = conn.cursor()

c.execute("SELECT id, file_path, status, result_json FROM jobs WHERE file_path LIKE '%CH-1 Electric Charges%'")
doc = c.fetchone()
if not doc:
    print("Job not found in DB")
else:
    job_id = doc[0]
    results = json.loads(doc[3]) if doc[3] else {}
    print(f"Job ID: {job_id}")
    questions = results.get("questions", [])
    print(json.dumps(questions[:3], indent=2))
