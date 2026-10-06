import fitz

doc = fitz.open("/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/12CB CH-3 Current Electricity (DPP).pdf")
for i, page in enumerate(doc):
    print(f"--- PAGE {i} ---")
    print(page.get_text())
