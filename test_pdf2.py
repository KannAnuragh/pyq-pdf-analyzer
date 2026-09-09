import sys
import os
import json
from ingest.router import route_and_extract

if __name__ == "__main__":
    pdf_path = "/app/pdf_qa_agent/data/uploads/59742498-ce96-42fe-b849-09a371f70b50_+2 CBSE -6 Application of Derivatives and multiconcepts.pdf"
    res = route_and_extract(pdf_path)
    print("Full text length:", len(res["full_text"]))
    
    # Save raw text to see if it extracted
    with open("/app/pdf_qa_agent/raw_extracted.txt", "w") as f:
        f.write(res["full_text"])
