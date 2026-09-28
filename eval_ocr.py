import json
import base64
import os
from ingest.structural_ocr import perform_structural_ocr, normalize_latex

def run_eval(eval_file="eval_crops.json"):
    if not os.path.exists(eval_file):
        print(f"Eval file {eval_file} not found. Please create it with base64 encoded crops.")
        return
        
    with open(eval_file, "r") as f:
        data = json.load(f)
        
    correct = 0
    total = len(data)
    if total == 0:
        print("No eval crops found.")
        return
        
    for item in data:
        img_bytes = base64.b64decode(item["image_b64"])
        expected = item["latex"]
        region_type = item.get("region_type", "fraction")
        
        predicted, ambiguous, review, p_tok, c_tok = perform_structural_ocr(img_bytes, region_type)
        
        if normalize_latex(predicted) == normalize_latex(expected):
            correct += 1
            print(f"PASS: {expected}")
        else:
            print(f"FAIL: Expected '{expected}', Got '{predicted}'")
            
    print(f"\nAccuracy: {correct}/{total} ({(correct/total)*100:.1f}%)")

if __name__ == "__main__":
    run_eval()
