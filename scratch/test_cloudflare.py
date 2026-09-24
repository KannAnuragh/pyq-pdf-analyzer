"""
Test script to verify Cloudflare R2 upload connectivity.

Usage:
    python scratch/test_cloudflare.py
"""
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path and load .env
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
load_dotenv(os.path.join(parent_dir, ".env"))

print("--- Cloudflare R2 Configuration Check ---")
account_id = os.getenv("R2_ACCOUNT_ID", "").strip()
access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
bucket = os.getenv("R2_BUCKET_NAME", "").strip()
domain = os.getenv("R2_CUSTOM_DOMAIN", "").strip()

print(f"R2_ACCOUNT_ID:        {'Found (' + account_id[:4] + '...' + account_id[-4:] + ')' if account_id else 'MISSING'}")
print(f"R2_ACCESS_KEY_ID:     {'Found (' + access_key[:4] + '...' + access_key[-4:] + ')' if access_key else 'MISSING'}")
print(f"R2_SECRET_ACCESS_KEY: {'Found (' + secret_key[:4] + '...' + secret_key[-4:] + ')' if secret_key else 'MISSING'}")
print(f"R2_BUCKET_NAME:       {bucket or 'MISSING'}")
print(f"R2_CUSTOM_DOMAIN:     {domain or 'MISSING'}")

missing = []
if not account_id: missing.append("R2_ACCOUNT_ID")
if not access_key: missing.append("R2_ACCESS_KEY_ID")
if not secret_key: missing.append("R2_SECRET_ACCESS_KEY")
if not bucket: missing.append("R2_BUCKET_NAME")

if missing:
    print(f"\n❌ Missing required env vars: {', '.join(missing)}")
    print("   Set them in .env and try again.")
    sys.exit(1)

print()

from storage.cloudflare import upload_image_to_cloudflare

# Test with a dummy 1x1 black pixel PNG
dummy_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

print("Uploading test image to R2...")
result = upload_image_to_cloudflare(dummy_image, filename="test-upload")

if result.startswith("https://"):
    print(f"\n✅ SUCCESS! Uploaded to Cloudflare R2:")
    print(f"   URL: {result}")
    print(f"\n   Try opening this URL in your browser to verify the custom domain serves it.")
elif result.startswith("data:image"):
    print(f"\n❌ FAILED (Fell back to base64).")
    print("   Check your R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and token permissions.")
    print("   The token needs 'Object Read & Write' permission on the bucket.")
else:
    print(f"\n⚠️  Unexpected result: {result[:100]}...")
