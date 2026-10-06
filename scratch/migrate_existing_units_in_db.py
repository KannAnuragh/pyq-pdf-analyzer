import sqlite3
import json
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.math_normalizer import normalize_math_in_text, heal_units_and_slashes

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pdf_qa.db")

def migrate_database():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, result_json FROM jobs")
    rows = cursor.fetchall()

    total_jobs_updated = 0
    total_questions_healed = 0

    for job_id, filename, r_json in rows:
        if not r_json:
            continue
        try:
            data = json.loads(r_json)
        except Exception:
            continue

        def _clean_obj(obj):
            if isinstance(obj, str):
                return normalize_math_in_text(obj)
            elif isinstance(obj, dict):
                return {k: _clean_obj(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean_obj(item) for item in obj]
            return obj

        cleaned_data = _clean_obj(data)
        if cleaned_data != data:
            new_r_json = json.dumps(cleaned_data, ensure_ascii=False)
            cursor.execute("UPDATE jobs SET result_json=? WHERE id=?", (new_r_json, job_id))
            total_jobs_updated += 1
            print(f"Healed job {job_id} ({filename})")

    conn.commit()
    conn.close()
    print(f"\nMigration complete: {total_jobs_updated} jobs updated, {total_questions_healed} questions healed.")

if __name__ == "__main__":
    migrate_database()
