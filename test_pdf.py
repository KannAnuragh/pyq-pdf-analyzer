import sys
import os
import json
from ingest.router import route_and_extract

if __name__ == "__main__":
    pdf_path = "/home/anuragh/Downloads/+2 CBSE -6 Application of Derivatives and multiconcepts.pdf"
    res = route_and_extract(pdf_path)
    with open("pdf_extracted_test.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Extracted successfully!")
