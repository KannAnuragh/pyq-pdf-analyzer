import os
import sys
import unittest
import json
import datetime
from fastapi.testclient import TestClient

from pdf_qa_agent.api.main import app
from pdf_qa_agent.storage import database, models
from pdf_qa_agent.api.auth import create_client_and_key
from pdf_qa_agent.api.cleanup import run_startup_cleanup


class TestV1Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.db = database.SessionLocal()

        # Create Client A
        cls.client_a, cls.key_a = create_client_and_key(
            db=cls.db,
            name="Test Client A (Primary)",
            daily_upload_limit=10
        )

        # Create Client B
        cls.client_b, cls.key_b = create_client_and_key(
            db=cls.db,
            name="Test Client B (Tenant B)",
            daily_upload_limit=10
        )

        # Create Client C with tight upload limit
        cls.client_c, cls.key_c = create_client_and_key(
            db=cls.db,
            name="Test Client C (Rate Limited)",
            daily_upload_limit=1
        )

        # Load dummy PDF bytes
        dummy_path = os.path.join(os.path.dirname(__file__), "..", "dummy.pdf")
        if os.path.exists(dummy_path):
            with open(dummy_path, "rb") as f:
                cls.pdf_bytes = f.read()
        else:
            cls.pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    @classmethod
    def tearDownClass(cls):
        import time, sys
        time.sleep(0.5)
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            test_clients = cls.db.query(models.Client).filter(
                models.Client.name.like("Test Client %") | models.Client.name.like("CLI Test %")
            ).all()
            for tc in test_clients:
                cls.db.query(models.ReviewFlag).filter(models.ReviewFlag.job_id.in_(
                    cls.db.query(models.Job.id).filter(models.Job.client_id == tc.id)
                )).delete(synchronize_session=False)
                cls.db.query(models.Job).filter(models.Job.client_id == tc.id).delete(synchronize_session=False)
                cls.db.query(models.ApiKey).filter(models.ApiKey.client_id == tc.id).delete(synchronize_session=False)
                cls.db.delete(tc)
            cls.db.commit()
        except Exception:
            pass
        finally:
            cls.db.close()

    def test_01_unauthenticated_requests_get_401(self):
        """Requests missing X-API-Key or with an invalid key must return 401 Unauthorized."""
        # 1. No key on V1 upload
        r1 = self.client.post("/v1/pdfs/upload")
        self.assertEqual(r1.status_code, 401)
        self.assertIn("Missing API key", r1.json()["detail"])

        # 2. No key on V1 status
        r2 = self.client.get("/v1/pdfs/some-fake-id/status")
        self.assertEqual(r2.status_code, 401)

        # 3. Invalid key
        r3 = self.client.get(
            "/v1/pdfs/some-fake-id/status",
            headers={"X-API-Key": "qscan_live_invalidkey1234567890"}
        )
        self.assertEqual(r3.status_code, 401)
        self.assertIn("Invalid or revoked", r3.json()["detail"])

        # 4. V1 list must reject unauthenticated callers
        r4 = self.client.get("/v1/pdfs")
        self.assertEqual(r4.status_code, 401)

        # 5. Keyless legacy requests must return 401
        r5 = self.client.get("/pdfs")
        self.assertEqual(r5.status_code, 401)
        self.assertIn("Missing API key", r5.json()["detail"])

        r6 = self.client.post("/pdfs/upload")
        self.assertEqual(r6.status_code, 401)

    def test_02_key_a_lifecycle_succeeds(self):
        """Client A uploads a PDF, checks status, receives 409 while processing, and 200 on completion."""
        # 1. Upload valid PDF
        resp_upload = self.client.post(
            "/v1/pdfs/upload",
            headers={"X-API-Key": self.key_a},
            files={"file": ("test_exam.pdf", self.pdf_bytes, "application/pdf")}
        )
        self.assertEqual(resp_upload.status_code, 202)
        data = resp_upload.json()
        job_id = data["job_id"]
        self.assertEqual(data["status"], "pending")
        self.assertEqual(data["filename"], "test_exam.pdf")

        # 2. Poll Status
        resp_status = self.client.get(
            f"/v1/pdfs/{job_id}/status",
            headers={"X-API-Key": self.key_a}
        )
        self.assertEqual(resp_status.status_code, 200)
        status_data = resp_status.json()
        self.assertEqual(status_data["job_id"], job_id)
        self.assertIn(status_data["status"], ["pending", "extracting", "correcting", "done"])

        # Wait briefly for background worker or explicitly check status
        # 3. Test that while status is not 'done', /result returns 409
        # Create a specific in-progress job for testing 409
        pending_job = models.Job(
            client_id=self.client_a.id,
            filename="in_progress.pdf",
            status="correcting",
            error_message="Fixing OCR typos"
        )
        self.db.add(pending_job)
        self.db.commit()

        resp_result_pending = self.client.get(
            f"/v1/pdfs/{pending_job.id}/result",
            headers={"X-API-Key": self.key_a}
        )
        self.assertEqual(resp_result_pending.status_code, 409)
        detail = resp_result_pending.json()["detail"]
        self.assertEqual(detail["job_id"], pending_job.id)
        self.assertEqual(detail["status"], "correcting")
        self.assertEqual(detail["error"], "Fixing OCR typos")

        # 4. Create a completed job and verify /result returns 200 with JSON payload
        sample_questions = [
            {
                "question_number": 1,
                "question_type": "mcq",
                "stem": "What is the unit of force?",
                "options": {"A": "Newton", "B": "Joule", "C": "Watt", "D": "Pascal"},
                "validated_answer": "A"
            }
        ]
        result_payload = {
            "source_file": "test_exam.pdf",
            "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "questions": sample_questions,
            "summary": {"total": 1, "validated": 1, "flagged": 0}
        }
        completed_job = models.Job(
            client_id=self.client_a.id,
            filename="completed_exam.pdf",
            status="done",
            result_json=json.dumps(result_payload)
        )
        self.db.add(completed_job)
        self.db.commit()

        # 5. Fetch Result when done -> 200 with JSON payload
        resp_result_done = self.client.get(
            f"/v1/pdfs/{completed_job.id}/result",
            headers={"X-API-Key": self.key_a}
        )
        self.assertEqual(resp_result_done.status_code, 200)
        self.assertEqual(resp_result_done.json()["source_file"], "test_exam.pdf")
        self.assertEqual(len(resp_result_done.json()["questions"]), 1)

    def test_03_key_b_cross_client_gets_404(self):
        """Client B requesting Client A's job_id must get 404 Not Found (no cross-tenant leakage)."""
        # Upload a job for Client A
        resp_upload = self.client.post(
            "/v1/pdfs/upload",
            headers={"X-API-Key": self.key_a},
            files={"file": ("client_a_secret.pdf", self.pdf_bytes, "application/pdf")}
        )
        job_id_a = resp_upload.json()["job_id"]

        # Client B attempts to fetch status -> 404
        r_status = self.client.get(
            f"/v1/pdfs/{job_id_a}/status",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(r_status.status_code, 404)
        self.assertEqual(r_status.json()["detail"], "Job not found")

        # Client B attempts to fetch result -> 404
        r_result = self.client.get(
            f"/v1/pdfs/{job_id_a}/result",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(r_result.status_code, 404)
        self.assertEqual(r_result.json()["detail"], "Job not found")

        # Client B attempts to delete Client A's job -> 404
        r_del = self.client.delete(
            f"/v1/pdfs/{job_id_a}",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(r_del.status_code, 404)

    def test_04_key_b_on_image_route_gets_404(self):
        """Client B requesting Client A's question image must return 404 Not Found."""
        # Setup Client A job with result JSON
        job_a = models.Job(
            client_id=self.client_a.id,
            filename="math_paper.pdf",
            file_path="/tmp/fake_nonexistent.pdf",
            status="done",
            result_json=json.dumps({
                "source_file": "math_paper.pdf",
                "questions": [{"question_number": 1, "stem": "Sample"}]
            })
        )
        self.db.add(job_a)
        self.db.commit()
        self.db.refresh(job_a)

        # Client B attempts to load question 1 image for Client A's job -> MUST return 404
        resp_img = self.client.get(
            f"/v1/pdfs/{job_a.id}/questions/1/image",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(resp_img.status_code, 404)
        self.assertEqual(resp_img.json()["detail"], "Job not found")

    def test_05_stream_upload_validation_and_size_cap(self):
        """Upload must reject files not starting with %PDF- and enforce size caps."""
        # 1. Reject non-PDF content (plain text)
        resp_bad = self.client.post(
            "/v1/pdfs/upload",
            headers={"X-API-Key": self.key_a},
            files={"file": ("notes.txt", b"Hello, this is just plain text notes.", "application/pdf")}
        )
        self.assertEqual(resp_bad.status_code, 400)
        self.assertIn("valid PDF starting with %PDF-", resp_bad.json()["detail"])

        # 2. Reject oversized file (simulated by setting environment cap to 500 bytes)
        orig_max = os.environ.get("MAX_UPLOAD_SIZE_BYTES")
        try:
            from pdf_qa_agent.api import v1_routes
            v1_routes.MAX_UPLOAD_SIZE_BYTES = 100  # 100 bytes limit for test

            big_content = b"%PDF-1.4 " + (b"A" * 500)
            resp_oversized = self.client.post(
                "/v1/pdfs/upload",
                headers={"X-API-Key": self.key_a},
                files={"file": ("huge.pdf", big_content, "application/pdf")}
            )
            self.assertEqual(resp_oversized.status_code, 413)
        finally:
            v1_routes.MAX_UPLOAD_SIZE_BYTES = int(orig_max or (50 * 1024 * 1024))

    def test_06_daily_upload_cap(self):
        """Client exceeding their daily upload cap must receive 429 Too Many Requests."""
        # Client C limit = 1
        # First upload -> 202
        r1 = self.client.post(
            "/v1/pdfs/upload",
            headers={"X-API-Key": self.key_c},
            files={"file": ("doc1.pdf", self.pdf_bytes, "application/pdf")}
        )
        self.assertEqual(r1.status_code, 202)

        # Second upload -> 429
        r2 = self.client.post(
            "/v1/pdfs/upload",
            headers={"X-API-Key": self.key_c},
            files={"file": ("doc2.pdf", self.pdf_bytes, "application/pdf")}
        )
        self.assertEqual(r2.status_code, 429)
        self.assertIn("Daily upload limit reached", r2.json()["detail"])

    def test_07_startup_cleanup_fails_stuck_and_requeues_pending(self):
        """Startup cleanup fails jobs stuck > 30 minutes, and keeps/re-queues pending ones."""
        now = datetime.datetime.utcnow()
        old_time = now - datetime.timedelta(minutes=45)
        recent_time = now - datetime.timedelta(minutes=10)

        dummy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dummy.pdf"))

        # Job 1: Stuck in 'extracting' 45 mins ago -> should become 'failed'
        job_stuck = models.Job(
            client_id=self.client_a.id,
            filename="stuck.pdf",
            file_path=dummy_path,
            status="extracting",
            created_at=old_time
        )

        # Job 2: Interrupted 10 mins ago -> should be reset to 'pending'
        job_interrupted = models.Job(
            client_id=self.client_a.id,
            filename="recent.pdf",
            file_path=dummy_path,
            status="correcting",
            created_at=recent_time
        )

        # Job 3: Legitimate pending job -> should stay 'pending'
        job_pending = models.Job(
            client_id=self.client_a.id,
            filename="pending.pdf",
            file_path=dummy_path,
            status="pending",
            created_at=now
        )

        self.db.add_all([job_stuck, job_interrupted, job_pending])
        self.db.commit()

        # Run startup cleanup
        run_startup_cleanup()

        # Re-query
        self.db.refresh(job_stuck)
        self.db.refresh(job_interrupted)
        self.db.refresh(job_pending)

        # Assertions
        self.assertEqual(job_stuck.status, "failed")
        self.assertIn("exceeded 30 minutes", job_stuck.error_message)

        self.assertIn(job_interrupted.status, ["pending", "extracting"])
        self.assertIn(job_pending.status, ["pending", "extracting"])

    def test_08_registration_endpoint_removed(self):
        """Phase 1: /v1/register has been removed from the public API; provisioning is CLI only."""
        bad_resp = self.client.post(
            "/v1/register",
            json={"name": "Attacker Client", "admin_token": "any"}
        )
        self.assertEqual(bad_resp.status_code, 404)

    def test_09_legacy_routes_two_keys_isolation(self):
        """Two different keys on legacy routes must NEVER see each other's jobs, and keyless requests must return 401."""
        # 1. Keyless request to legacy routes -> 401
        r_unauth_list = self.client.get("/pdfs")
        self.assertEqual(r_unauth_list.status_code, 401)

        r_unauth_upload = self.client.post("/pdfs/upload")
        self.assertEqual(r_unauth_upload.status_code, 401)

        # 2. Client A uploads a job via legacy /pdfs/upload
        resp_a = self.client.post(
            "/pdfs/upload",
            headers={"X-API-Key": self.key_a},
            files={"file": ("client_a_legacy.pdf", self.pdf_bytes, "application/pdf")}
        )
        self.assertEqual(resp_a.status_code, 200)
        job_a_id = resp_a.json()["job_id"]

        # 3. Client A lists legacy /pdfs -> Sees its job
        resp_a_list = self.client.get(
            "/pdfs",
            headers={"X-API-Key": self.key_a}
        )
        self.assertEqual(resp_a_list.status_code, 200)
        a_job_ids = [j["job_id"] for j in resp_a_list.json()]
        self.assertIn(job_a_id, a_job_ids)

        # 4. Client B lists legacy /pdfs -> MUST NOT see Client A's job
        resp_b_list = self.client.get(
            "/pdfs",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(resp_b_list.status_code, 200)
        b_job_ids = [j["job_id"] for j in resp_b_list.json()]
        self.assertNotIn(job_a_id, b_job_ids)

        # 5. Client B requests Client A's job status on legacy route -> 404
        r_b_status = self.client.get(
            f"/pdfs/{job_a_id}/status",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(r_b_status.status_code, 404)

        # 6. Client B requests Client A's job result on legacy route -> 404
        r_b_result = self.client.get(
            f"/pdfs/{job_a_id}/result",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(r_b_result.status_code, 404)

        # 7. Client B requests Client A's job file on legacy route -> 404
        r_b_file = self.client.get(
            f"/pdfs/{job_a_id}/file",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(r_b_file.status_code, 404)

        # 8. Client B requests Client A's question image on legacy route -> 404
        r_b_img = self.client.get(
            f"/pdfs/{job_a_id}/questions/1/image",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(r_b_img.status_code, 404)

        # 9. Client B attempts to delete Client A's job on legacy route -> 404
        r_b_del = self.client.delete(
            f"/pdfs/{job_a_id}",
            headers={"X-API-Key": self.key_b}
        )
        self.assertEqual(r_b_del.status_code, 404)

    def test_10_legacy_routes_internal_network_restriction(self):
        """When RESTRICT_LEGACY_TO_INTERNAL is true, legacy routes block external IPs (403) while allowing internal IPs."""
        orig_restrict = os.environ.get("RESTRICT_LEGACY_TO_INTERNAL")
        os.environ["RESTRICT_LEGACY_TO_INTERNAL"] = "true"

        try:
            # 1. External IP (e.g. public IP 93.184.216.34) attempting to access legacy routes -> 403 Forbidden
            r_external = self.client.get(
                "/pdfs",
                headers={
                    "X-API-Key": self.key_a,
                    "X-Forwarded-For": "93.184.216.34"
                }
            )
            self.assertEqual(r_external.status_code, 403)
            self.assertIn("Network restricted to internal callers", r_external.json()["detail"])

            # 2. Internal / Private IP (e.g. 10.0.0.12 or 172.18.0.2 or 192.168.1.50) -> 200 OK
            r_internal = self.client.get(
                "/pdfs",
                headers={
                    "X-API-Key": self.key_a,
                    "X-Forwarded-For": "10.0.0.12"
                }
            )
            self.assertEqual(r_internal.status_code, 200)

            # 3. Public V1 routes remain accessible externally even if legacy is restricted
            r_v1 = self.client.get(
                "/v1/pdfs",
                headers={
                    "X-API-Key": self.key_a,
                    "X-Forwarded-For": "93.184.216.34"
                }
            )
            self.assertEqual(r_v1.status_code, 200)

        finally:
            if orig_restrict is not None:
                os.environ["RESTRICT_LEGACY_TO_INTERNAL"] = orig_restrict
            else:
                os.environ.pop("RESTRICT_LEGACY_TO_INTERNAL", None)

    def test_11_client_disabled_makes_active_key_return_401(self):
        """Phase 3 Checkpoint: Disabling a client makes its still-active key return 401."""
        # Client A is active and key works
        resp_ok = self.client.get("/v1/pdfs", headers={"X-API-Key": self.key_a})
        self.assertEqual(resp_ok.status_code, 200)

        # Disable Client A in database (while key_a itself remains is_active=True)
        self.client_a.is_active = False
        self.db.commit()

        # Key A must now return 401 Unauthorized
        resp_disabled = self.client.get("/v1/pdfs", headers={"X-API-Key": self.key_a})
        self.assertEqual(resp_disabled.status_code, 401)
        self.assertIn("inactive", resp_disabled.json()["detail"].lower())

        # Re-enable Client A
        self.client_a.is_active = True
        self.db.commit()

        # Key A works again
        resp_restored = self.client.get("/v1/pdfs", headers={"X-API-Key": self.key_a})
        self.assertEqual(resp_restored.status_code, 200)

    def test_12_admin_cli_lifecycle(self):
        """Phase 5 Checkpoint: CLI create client, key works, disable -> 401, enable restores, unknown ID exits nonzero."""
        import subprocess

        # 1. Create client via CLI
        cli_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "admin_cli.py")
        res_create = subprocess.run(
            [sys.executable, cli_py, "clients", "create", "CLI Test Tenant", "--limit", "50"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_create.returncode, 0)
        self.assertIn("Client successfully provisioned", res_create.stdout)

        # Extract Client ID
        import re
        m = re.search(r"ID:\s+([a-zA-Z0-9_-]+)", res_create.stdout)
        self.assertTrue(m is not None)
        cli_client_id = m.group(1)

        # 2. Create Key via CLI
        res_key = subprocess.run(
            [sys.executable, cli_py, "keys", "create", cli_client_id, "--name", "CLI Key"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_key.returncode, 0)
        km = re.search(r"RAW API KEY:\s+(qscan_live_[a-zA-Z0-9_-]+)", res_key.stdout)
        self.assertTrue(km is not None)
        cli_raw_key = km.group(1)

        # 3. Key works against API -> 200
        r_test_ok = self.client.get("/v1/pdfs", headers={"X-API-Key": cli_raw_key})
        self.assertEqual(r_test_ok.status_code, 200)

        # 4. Disable client via CLI -> key returns 401
        res_dis = subprocess.run(
            [sys.executable, cli_py, "clients", "disable", cli_client_id, "-y"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_dis.returncode, 0)

        r_test_401 = self.client.get("/v1/pdfs", headers={"X-API-Key": cli_raw_key})
        self.assertEqual(r_test_401.status_code, 401)

        # 5. Enable client via CLI -> key restored to 200
        res_en = subprocess.run(
            [sys.executable, cli_py, "clients", "enable", cli_client_id, "-y"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_en.returncode, 0)

        r_test_restored = self.client.get("/v1/pdfs", headers={"X-API-Key": cli_raw_key})
        self.assertEqual(r_test_restored.status_code, 200)

        # 6. Unknown client ID exits nonzero (returncode != 0)
        res_unknown = subprocess.run(
            [sys.executable, cli_py, "clients", "disable", "nonexistent_fake_id_12345", "-y"],
            capture_output=True,
            text=True
        )
        self.assertNotEqual(res_unknown.returncode, 0)

        # 7. Internal client protection: disabling c_admin without --force exits nonzero
        res_internal = subprocess.run(
            [sys.executable, cli_py, "clients", "disable", "c_admin", "-y"],
            capture_output=True,
            text=True
        )
        self.assertNotEqual(res_internal.returncode, 0)
        self.assertIn("without --force", res_internal.stderr)


if __name__ == "__main__":
    unittest.main()

