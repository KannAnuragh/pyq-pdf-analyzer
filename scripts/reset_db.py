#!/usr/bin/env python3
"""
Complete Database Reset Script
Wipes all clients, API keys, jobs, and review flags.
Reinitializes clean accounts for:
  - c_admin (Super Admin)
  - c_frontend_1 (Portal Frontend)
  - c_frontend_2 (Studio Frontend)
"""

import sys
import os
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_root)

from pdf_qa_agent.storage import database, models

def main():
    print("=" * 70)
    print("PDF QA Agent - Database Reset & Clean Re-initialization")
    print("=" * 70)

    db = database.SessionLocal()
    try:
        # 1. Clear all existing rows
        print("Clearing ReviewFlags...")
        db.query(models.ReviewFlag).delete()
        print("Clearing Jobs...")
        db.query(models.Job).delete()
        print("Clearing ApiKeys...")
        db.query(models.ApiKey).delete()
        print("Clearing Clients...")
        db.query(models.Client).delete()
        db.commit()

        # 2. Clear old uploaded files on disk
        uploads_dir = getattr(database, "UPLOADS_DIR", os.path.join(project_root, "data", "uploads"))
        if os.path.exists(uploads_dir):
            for item in os.listdir(uploads_dir):
                item_path = os.path.join(uploads_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    print(f"Notice cleaning upload file {item_path}: {e}")
        os.makedirs(uploads_dir, exist_ok=True)

        print("\nAll previous clients, keys, jobs, and uploads wiped.")
    finally:
        db.close()

    # 3. Re-run init_db to provision c_admin
    database.init_db()

    # 4. Display newly provisioned accounts
    db = database.SessionLocal()
    try:
        clients = db.query(models.Client).order_by(models.Client.id.asc()).all()
        keys = db.query(models.ApiKey).order_by(models.ApiKey.client_id.asc()).all()

        print("\n" + "=" * 90)
        print("Active Accounts Successfully Initialized:")
        print("=" * 90)
        print(f"{'CLIENT ID':<16} | {'NAME':<24} | {'ACTIVE':<7} | {'ADMIN':<6} | {'DAILY LIMIT'}")
        print("-" * 90)
        for c in clients:
            print(f"{c.id:<16} | {c.name:<24} | {'YES' if c.is_active else 'NO':<7} | {'YES' if c.is_admin else 'NO':<6} | {c.daily_upload_limit}")
        print("=" * 90)

        print("\nMaster Admin API Key:")
        admin_key = os.getenv("ADMIN_API_KEY", "qscan_live_admin_master_key_2026")
        print(f"  c_admin (Admin Master):  {admin_key}")
        print("\nAdd frontend or client tenants when needed via CLI:")
        print("  python -m pdf_qa_agent.scripts.admin_cli clients create --name \"Frontend\"")
        print("=" * 90 + "\n")
    finally:
        db.close()

if __name__ == "__main__":
    main()
