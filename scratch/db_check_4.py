import sqlite3
import json

conn = sqlite3.connect('/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/data/pdf_qa.db')
cursor = conn.cursor()

cursor.execute("SELECT id, result_json FROM jobs LIMIT 10")
rows = cursor.fetchall()
found = False
for row in rows:
    data = json.loads(row[1])
    for q in data.get('questions', []):
        if 'TAG' in str(q):
            print(f"Job {row[0]}, Q{q.get('question_number')}:")
            print(f"stem: {q.get('stem', '')[:100]}")
            print(f"explanation: {q.get('explanation', '')[:100]}")
            found = True
if not found:
    print("No job with TAG found.")
conn.close()
