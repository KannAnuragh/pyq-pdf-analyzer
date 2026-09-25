import sqlite3

conn = sqlite3.connect('/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/data/pdf_qa.db')
cursor = conn.cursor()

# Get the last processed question that has HTMLTAG in its explanation or stem
cursor.execute("SELECT question_number, corrected_stem, corrected_explanation FROM trusted_corrections WHERE corrected_explanation LIKE '%HTMLTAG%' OR corrected_stem LIKE '%HTMLTAG%' LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(f"Q{row[0]}:")
    print(f"  corrected_stem: {repr(row[1])[:200]}")
    print(f"  corrected_explanation: {repr(row[2])[:200]}")
    print("-" * 50)

conn.close()
