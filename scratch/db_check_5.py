import sqlite3

conn = sqlite3.connect('/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/data/pdf_qa.db')
cursor = conn.cursor()

cursor.execute("SELECT id, result_json FROM jobs WHERE result_json LIKE '%HTMLTAG%'")
rows = cursor.fetchall()
if not rows:
    print("No HTMLTAG found in raw DB.")
else:
    for row in rows:
        print(f"Job {row[0]} has HTMLTAG")
conn.close()
