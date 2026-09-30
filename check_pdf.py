import sqlite3
import fitz
import sys
conn = sqlite3.connect('sql_app.db')
c = conn.cursor()
c.execute("SELECT file_path FROM documents LIMIT 1")
row = c.fetchone()
if not row:
    print("No docs")
    sys.exit()
doc = fitz.open(row[0])
for i, page in enumerate(doc):
    text = page.get_text()
    if "Why do equipotential surfaces" in text or "close to each other" in text:
        print(f"--- PAGE {i} ---")
        print(text)
