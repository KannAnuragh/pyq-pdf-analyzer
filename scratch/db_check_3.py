import sqlite3
import json

conn = sqlite3.connect('/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/data/pdf_qa.db')
cursor = conn.cursor()

cursor.execute("SELECT id, result_json FROM jobs WHERE result_json LIKE '%HTML%' LIMIT 1")
row = cursor.fetchone()
if row:
    print(f"Job {row[0]}:")
    data = json.loads(row[1])
    for q in data.get('questions', []):
        if 'HTML' in str(q):
            print(f"Q{q.get('question_number')}:")
            print(f"stem: {q.get('stem', '')[:100]}")
            print(f"explanation: {q.get('explanation', '')[:100]}")
            print(f"corrected_explanation: {q.get('corrected_explanation', '')[:100]}")
            print("-" * 50)
else:
    print("No job with HTML found.")
conn.close()
