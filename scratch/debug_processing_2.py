import sys
from ingest.ocr_word_fallback import resolve_ocr_queue
from storage import database

db = database.SessionLocal()
try:
    print("Testing OCR queue...")
    # Just run on a sample file
    resolve_ocr_queue(db, "test_job_1", "data/uploads/15d82dbb-c7ec-4a23-81c3-0993e01a3bb1_test_carpen.pdf")
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    db.close()
