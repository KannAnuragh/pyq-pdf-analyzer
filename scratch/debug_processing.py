import sys
from ingest.router import route_and_extract
from storage import database

db = database.SessionLocal()
try:
    print("Testing processing...")
    # Just run on a sample file
    res = route_and_extract("data/uploads/15d82dbb-c7ec-4a23-81c3-0993e01a3bb1_test_carpen.pdf", "test_job_1", db)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    db.close()
