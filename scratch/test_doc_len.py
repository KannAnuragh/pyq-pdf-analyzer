import sqlite3
import json
import pymupdf

db = sqlite3.connect("data/pdf_qa.db")
c = db.cursor()
c.execute("SELECT id, pdf_path FROM jobs ORDER BY created_at DESC LIMIT 1")
job_id, pdf_path = c.fetchone()
print(f"PDF Path: {pdf_path}")
doc = pymupdf.open(pdf_path)
print(f"Pages: {len(doc)}")
