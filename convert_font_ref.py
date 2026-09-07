import sys
import os
import time
import io
import fitz  # PyMuPDF
from PIL import Image
from google import genai
from google.genai import types

def convert_pdf_with_gemini(pdf_path: str, output_txt_path: str):
    print(f"Opening PDF: {pdf_path}")
    
    # 1. Check API Key
    # Ask user for key if not in env
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API Key (starts with AIzaSy...): ").strip()
        if not api_key:
            print("API Key is required!")
            sys.exit(1)
            
    client = genai.Client(api_key=api_key)
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        sys.exit(1)
        
    total_pages = len(doc)
    
    # Open the text file in append mode so we don't lose progress if it crashes!
    print(f"\nStarting Gemini OCR Conversion. Progress will be saved directly to: {output_txt_path}")
    print("This will take about 4.5 seconds per page to respect free-tier rate limits.")
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(doc, 1):
            print(f"Scanning page {i}/{total_pages}...", flush=True)
            
            # Render page as image (zoom=1.2 balances quality and upload speed)
            zoom = 1.2
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Rate Limiting (15 RPM free tier = 1 request every 4 seconds)
            if i > 1:
                time.sleep(4.5)
                
            try:
                # Call Gemini 2.0 Flash Vision
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[
                        types.Part.from_bytes(
                            data=img_data,
                            mime_type="image/png"
                        ),
                        "Extract all Malayalam and English text from this image. Return only the extracted text, preserving the reading order. Fix any legacy fonts. Do not write any intro or outro text."
                    ]
                )
                
                text = response.text.strip()
                
                # Apply SMC post-processing to clean any leftover FML legacy font symbols (≥, ‰, ƒ, ∏, ™, etc.)
                try:
                    from backend.ingestion.smc_fml_converter import convert_fml_to_unicode
                    text = convert_fml_to_unicode(text)
                except ImportError:
                    try:
                        from ingestion.smc_fml_converter import convert_fml_to_unicode
                        text = convert_fml_to_unicode(text)
                    except Exception:
                        pass
                        
                f.write(f"{text}\n\n")
                f.flush() # Ensure it's saved immediately
                
            except Exception as e:
                print(f"❌ Gemini API failed on page {i}: {e}")
                print("Don't worry, you can restart the script later or fix this page manually.")
                f.write(f"\n[OCR FAILED FOR PAGE {i}]\n\n")
                
            # Cleanup memory
            del pix, img_data
            
    doc.close()
    
    print("\n✅ Conversion complete!")
    print(f"You can now upload '{os.path.basename(output_txt_path)}' directly to the admin dashboard!")
    print("Since it is a .txt file, the server will ingest it instantly with no errors.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_font.py <path_to_pdf>")
        print("Example: python convert_font.py textbook.pdf")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
        
    base_name = os.path.splitext(pdf_path)[0]
    output_txt_path = f"{base_name}_unicode.txt"
    
    convert_pdf_with_gemini(pdf_path, output_txt_path)
