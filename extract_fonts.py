import pymupdf
import sys
import os

# Ensure ingest module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.font_installer import install_reference_font, ensure_storage_dirs

def extract_and_install_fonts(pdf_path):
    print(f"Opening PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    ensure_storage_dirs()
    
    extracted_xrefs = set()
    
    for i in range(len(doc)):
        font_list = doc.get_page_fonts(i)
        for font_info in font_list:
            xref = font_info[0]
            if xref not in extracted_xrefs:
                extracted_xrefs.add(xref)
                font_name = font_info[3]
                font_data = doc.extract_font(xref)
                ext = font_data[1] if isinstance(font_data, tuple) and len(font_data) >= 3 else font_data[0] # Try to get ext
                buffer = font_data[-1]
                ext = font_data[1] if isinstance(font_data, tuple) and len(font_data) >= 3 else "ttf"
                if isinstance(font_data, tuple) and len(font_data) >= 4:
                    ext = font_data[1] # basename, ext, ext_type, buffer
                    buffer = font_data[3]
                elif isinstance(font_data, tuple) and len(font_data) == 3:
                    ext = font_data[1] # basename, ext, buffer
                    buffer = font_data[2]
                if ext == "ttf" or ext == "otf":
                    filename = f"{font_name}.{ext}"
                    print(f"Extracting and installing font: {filename} (xref: {xref})")
                    result = install_reference_font(buffer, filename)
                    print(f"Result: {result}")
                else:
                    print(f"Skipping non-TTF/OTF font: {font_name} (ext: {ext})")
    
    print(f"Done. Processed {len(extracted_xrefs)} fonts.")

if __name__ == "__main__":
    pdf_path = "/app/pdf_qa_agent/data/uploads/7147490e-8229-4896-81ed-286afb9622a1_CH-1 Electric Charges & Field (DPP).pdf"
    extract_and_install_fonts(pdf_path)
