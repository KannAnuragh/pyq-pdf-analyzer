import sqlite3
import json

conn = sqlite3.connect('/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/data/pdf_qa.db')
cursor = conn.cursor()

cursor.execute("SELECT id, result_json FROM jobs ORDER BY id DESC LIMIT 10")
rows = cursor.fetchall()
for row in rows:
    data = json.loads(row[1])
    for q in data.get('questions', []):
        if 'reactance' in str(q):
            print(repr(q.get('explanation')))
conn.close()
