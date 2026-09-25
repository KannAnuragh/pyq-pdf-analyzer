import sqlite3
import json

conn = sqlite3.connect('/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/data/pdf_qa.db')
cursor = conn.cursor()

cursor.execute("SELECT id, result_json FROM jobs")
rows = cursor.fetchall()
found = False
for row in rows:
    if '\u0002' in row[1] or '\uE000' in row[1] or 'HTMLTAG' in row[1]:
        print(f"Job {row[0]} has strange char.")
        found = True
if not found:
    print("Nothing strange found in raw DB.")
conn.close()
