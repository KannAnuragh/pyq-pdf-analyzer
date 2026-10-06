import sys
import os
import json

sys.path.append(os.path.abspath("."))
from ingest.router import process_pdf_document

pdf_path = "data/uploads/a71f0ab6-e3f0-4177-9d42-32e3f3947bcd_CH-1 Electric Charges & Field (DPP).pdf"
job_id = "test_job_1"

# Since we don't have the full DB setup or redis here, we can just call process_pdf_document.
# Wait, process_pdf_document writes to DB. Let's just use LangGraph directly or run it via the backend API.
