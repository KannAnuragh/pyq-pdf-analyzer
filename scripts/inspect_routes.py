import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_root)

from pdf_qa_agent.api.main import app
from pdf_qa_agent.api.auth import get_current_client, get_current_admin_client
from fastapi.routing import APIRoute

def collect_routes(routes, parent_prefix="", parent_deps=None):
    if parent_deps is None:
        parent_deps = []
    all_endpoints = []
    for r in routes:
        if isinstance(r, APIRoute):
            all_endpoints.append((r, parent_prefix + r.path, parent_deps + getattr(r, "dependencies", [])))
        elif hasattr(r, "original_router"):
            prefix = parent_prefix + (r.include_context.prefix if hasattr(r, "include_context") else "")
            inc_deps = parent_deps + (r.include_context.dependencies if hasattr(r, "include_context") else [])
            all_endpoints.extend(collect_routes(r.original_router.routes, prefix, inc_deps))
    return all_endpoints

endpoints = collect_routes(app.routes)

print("=" * 110)
print(f"{'METHOD':<8} | {'PATH':<55} | {'AUTH TYPE':<18} | {'STATUS'}")
print("=" * 110)

all_protected = True
sensitive_protected = True

for route, full_path, router_deps in endpoints:
    methods = ",".join(route.methods - {"HEAD", "OPTIONS"})
    dep_calls = [d.call for d in router_deps if hasattr(d, "call")]

    # Endpoint-level dependencies
    for d in route.dependant.dependencies:
        if hasattr(d, "call"):
            dep_calls.append(d.call)

    has_client_auth = any(d in (get_current_client, get_current_admin_client) for d in dep_calls)
    has_admin_auth = any(d == get_current_admin_client for d in dep_calls)

    status = "OK"
    if not has_client_auth:
        status = "FAIL: NO AUTH"
        all_protected = False

    auth_label = "ADMIN ONLY" if has_admin_auth else ("CLIENT AUTH" if has_client_auth else "NO AUTH")

    # Sensitive config / font upload
    if "/config" in full_path and "POST" in route.methods:
        if not has_admin_auth:
            status = "FAIL: POST /config NOT ADMIN"
            sensitive_protected = False
    if "/fonts/upload" in full_path and "POST" in route.methods:
        if not has_admin_auth:
            status = "FAIL: /fonts/upload NOT ADMIN"
            sensitive_protected = False

    print(f"{methods:<8} | {full_path:<55} | {auth_label:<18} | {status}")

print("=" * 110)
if all_protected and sensitive_protected:
    print(f"CHECKPOINT PASSED: All {len(endpoints)} application routes have strict authentication dependencies.")
    print("Administrative / config-altering routes strictly require get_current_admin_client.")
    sys.exit(0)
else:
    print("CHECKPOINT FAILED: One or more routes lack required protection.")
    sys.exit(1)
