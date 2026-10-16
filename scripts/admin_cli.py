#!/usr/bin/env python3
"""
Administrative CLI for PDF QA Agent
Multi-tenant client and API key provisioning and management.

Usage:
  python -m pdf_qa_agent.scripts.admin_cli clients list
  python -m pdf_qa_agent.scripts.admin_cli clients create [--name] "Client Name" [--daily-limit 100] [--admin]
  python -m pdf_qa_agent.scripts.admin_cli clients disable <client_id> [--force] [-y]
  python -m pdf_qa_agent.scripts.admin_cli clients enable <client_id> [--force] [-y]
  python -m pdf_qa_agent.scripts.admin_cli clients limit <client_id> <limit> [--force] [-y]
  python -m pdf_qa_agent.scripts.admin_cli clients delete <client_id> [--force] [-y]
  python -m pdf_qa_agent.scripts.admin_cli keys list [--client-id <client_id>]
  python -m pdf_qa_agent.scripts.admin_cli keys create [--client-id] <client_id> [--key "<RAW_KEY>"] [--name "Key Name"] [--create-client]
  python -m pdf_qa_agent.scripts.admin_cli keys add <client_id> <raw_key> [--name "Key Name"] [--create-client]
  python -m pdf_qa_agent.scripts.admin_cli keys disable <key_id> [-y]
  python -m pdf_qa_agent.scripts.admin_cli keys enable <key_id> [-y]
  python -m pdf_qa_agent.scripts.admin_cli keys delete <key_id> [-y]
"""

import sys
import os
import uuid
import argparse
import datetime

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
parent_root = os.path.dirname(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if parent_root not in sys.path:
    sys.path.insert(0, parent_root)

try:
    from pdf_qa_agent.storage import database, models
    from pdf_qa_agent.api.auth import generate_api_key
except ImportError:
    from storage import database, models
    from api.auth import generate_api_key


def log_audit(action: str, target: str, details: str = "") -> None:
    """Prints a standard audit line and appends it to data/audit.log."""
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    audit_line = f"[AUDIT {timestamp}] ACTION={action} TARGET={target} {details}".strip()
    print(audit_line)

    try:
        data_dir = getattr(database, "DATA_DIR", os.path.join(project_root, "data"))
        os.makedirs(data_dir, exist_ok=True)
        log_file = os.path.join(data_dir, "audit.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(audit_line + "\n")
    except Exception:
        pass


def confirm_prompt(prompt_msg: str, auto_confirm: bool) -> bool:
    """Prompts the operator for y/N confirmation unless auto_confirm is True."""
    if auto_confirm:
        return True
    try:
        reply = input(f"{prompt_msg} [y/N]: ").strip().lower()
        return reply in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# --- Client Subcommands ---

def handle_clients_list(args):
    db = database.SessionLocal()
    try:
        clients = db.query(models.Client).order_by(models.Client.created_at.desc()).all()
        if not clients:
            print("No clients found.")
            return

        print(f"{'CLIENT ID':<36} | {'NAME':<24} | {'ACTIVE':<7} | {'ADMIN':<6} | {'DAILY LIMIT':<11} | {'CREATED AT'}")
        print("-" * 115)
        for c in clients:
            created_str = c.created_at.strftime("%Y-%m-%d %H:%M") if c.created_at else "-"
            active_str = "YES" if c.is_active else "NO"
            admin_str = "YES" if c.is_admin else "NO"
            limit_str = "UNLIMITED" if (c.daily_upload_limit is None or c.daily_upload_limit <= 0 or c.daily_upload_limit >= 1000000) else str(c.daily_upload_limit)
            print(f"{c.id:<36} | {c.name[:24]:<24} | {active_str:<7} | {admin_str:<6} | {limit_str:<11} | {created_str}")
    finally:
        db.close()


def handle_clients_create(args):
    db = database.SessionLocal()
    try:
        name = args.flag_name or args.name
        if not name:
            print("Error: Client name must be specified via positional argument or --name.", file=sys.stderr)
            sys.exit(1)

        custom_id = getattr(args, "id", None)
        if custom_id:
            existing = db.query(models.Client).filter(models.Client.id == custom_id).first()
            if existing:
                print(f"Error: Client with ID '{custom_id}' already exists.", file=sys.stderr)
                sys.exit(1)

        client = models.Client(
            id=custom_id if custom_id else str(uuid.uuid4()),
            name=name,
            daily_upload_limit=args.limit,
            is_admin=args.admin,
            is_active=True
        )
        db.add(client)
        db.commit()
        db.refresh(client)

        limit_display = "UNLIMITED" if (client.daily_upload_limit is None or client.daily_upload_limit <= 0 or client.daily_upload_limit >= 1000000) else str(client.daily_upload_limit)
        log_audit("client_create", client.id, f"NAME='{client.name}' ADMIN={client.is_admin} LIMIT={limit_display}")
        print("\nClient successfully provisioned:")
        print(f"  ID:           {client.id}")
        print(f"  Name:         {client.name}")
        print(f"  Admin:        {client.is_admin}")
        print(f"  Daily Limit:  {limit_display}")
        print(f"\nNext step: Generate an API key using:")
        print(f"  python -m pdf_qa_agent.scripts.admin_cli keys create {client.id}\n")
    finally:
        db.close()


def handle_clients_disable(args):
    db = database.SessionLocal()
    try:
        client = db.query(models.Client).filter(models.Client.id == args.client_id).first()
        if not client:
            print(f"Error: Client '{args.client_id}' not found.", file=sys.stderr)
            sys.exit(1)

        # Protect master admin
        if client.id == "c_admin" and not args.force:
            print(f"Error: Refusing to disable master admin '{client.id}' without --force.", file=sys.stderr)
            sys.exit(1)

        if not client.is_active:
            print(f"Client '{client.id}' is already inactive.")
            return

        if not confirm_prompt(f"Are you sure you want to disable client '{client.name}' ({client.id})?", args.yes or args.force):
            print("Operation aborted by user.")
            sys.exit(0)

        client.is_active = False
        db.commit()
        log_audit("client_disable", client.id, f"NAME='{client.name}'")
        print(f"Client '{client.id}' disabled successfully.")
    finally:
        db.close()


def handle_clients_enable(args):
    db = database.SessionLocal()
    try:
        client = db.query(models.Client).filter(models.Client.id == args.client_id).first()
        if not client:
            print(f"Error: Client '{args.client_id}' not found.", file=sys.stderr)
            sys.exit(1)

        if client.is_active:
            print(f"Client '{client.id}' is already active.")
            return

        if not confirm_prompt(f"Are you sure you want to enable client '{client.name}' ({client.id})?", args.yes or args.force):
            print("Operation aborted by user.")
            sys.exit(0)

        client.is_active = True
        db.commit()
        log_audit("client_enable", client.id, f"NAME='{client.name}'")
        print(f"Client '{client.id}' enabled successfully.")
    finally:
        db.close()


def handle_clients_limit(args):
    db = database.SessionLocal()
    try:
        client = db.query(models.Client).filter(models.Client.id == args.client_id).first()
        if not client:
            print(f"Error: Client '{args.client_id}' not found.", file=sys.stderr)
            sys.exit(1)

        new_limit = None if (args.limit is None or args.limit <= 0) else args.limit
        old_display = "UNLIMITED" if (client.daily_upload_limit is None or client.daily_upload_limit <= 0 or client.daily_upload_limit >= 1000000) else str(client.daily_upload_limit)
        new_display = "UNLIMITED" if new_limit is None else str(new_limit)

        if not confirm_prompt(f"Change upload limit for '{client.name}' from {old_display} to {new_display}?", args.yes or args.force):
            print("Operation aborted by user.")
            sys.exit(0)

        client.daily_upload_limit = new_limit
        db.commit()
        log_audit("client_limit_change", client.id, f"OLD_LIMIT={old_display} NEW_LIMIT={new_display}")
        print(f"Daily limit for '{client.id}' updated to {new_display}.")
    finally:
        db.close()


def handle_clients_delete(args):
    db = database.SessionLocal()
    try:
        client = db.query(models.Client).filter(models.Client.id == args.client_id).first()
        if not client:
            print(f"Error: Client '{args.client_id}' not found.", file=sys.stderr)
            sys.exit(1)

        if client.id == "c_admin" and not args.force:
            print("Error: Refusing to delete master admin 'c_admin' without --force.", file=sys.stderr)
            sys.exit(1)

        if not confirm_prompt(f"PERMANENT ACTION: Delete client '{client.name}' ({client.id}) and all its keys/jobs?", args.yes or args.force):
            print("Operation aborted by user.")
            sys.exit(0)

        # Delete associated review flags, jobs, and api keys
        job_ids = [j.id for j in db.query(models.Job.id).filter(models.Job.client_id == client.id).all()]
        if job_ids:
            db.query(models.ReviewFlag).filter(models.ReviewFlag.job_id.in_(job_ids)).delete(synchronize_session=False)
            db.query(models.Job).filter(models.Job.client_id == client.id).delete(synchronize_session=False)

        db.query(models.ApiKey).filter(models.ApiKey.client_id == client.id).delete(synchronize_session=False)
        db.delete(client)
        db.commit()

        log_audit("client_delete", client.id, f"NAME='{client.name}'")
        print(f"Client '{client.id}' ({client.name}) and all associated data permanently deleted.")
    finally:
        db.close()


# --- Key Subcommands ---

def handle_keys_list(args):
    db = database.SessionLocal()
    try:
        query = db.query(models.ApiKey)
        if args.client_id:
            query = query.filter(models.ApiKey.client_id == args.client_id)
        keys = query.order_by(models.ApiKey.created_at.desc()).all()

        if not keys:
            print("No API keys found.")
            return

        print(f"{'KEY ID':<36} | {'CLIENT ID':<20} | {'NAME':<20} | {'PREFIX':<14} | {'ACTIVE':<7} | {'LAST USED':<16} | {'CREATED AT'}")
        print("-" * 135)
        for k in keys:
            created_str = k.created_at.strftime("%Y-%m-%d %H:%M") if k.created_at else "-"
            used_str = k.last_used_at.strftime("%Y-%m-%d %H:%M") if k.last_used_at else "Never"
            active_str = "YES" if k.is_active else "NO"
            print(f"{k.id:<36} | {k.client_id:<20} | {(k.name or '')[:20]:<20} | {(k.prefix or '')[:14]:<14} | {active_str:<7} | {used_str:<16} | {created_str}")
    finally:
        db.close()


def handle_keys_create(args):
    db = database.SessionLocal()
    try:
        client_id = getattr(args, "flag_client_id", None) or getattr(args, "client_id", None)
        if not client_id:
            print("Error: Client ID must be specified via positional argument or --client-id.", file=sys.stderr)
            sys.exit(1)

        client = db.query(models.Client).filter(models.Client.id == client_id).first()
        if not client:
            # Fallback: check if client_id was intended as client name
            client = db.query(models.Client).filter(models.Client.name == client_id).first()

        if not client:
            if getattr(args, "create_client", False):
                client = models.Client(
                    id=client_id,
                    name=getattr(args, "name", None) or f"Client {client_id}",
                    daily_upload_limit=100,
                    is_active=True
                )
                db.add(client)
                db.commit()
                db.refresh(client)
                log_audit("client_create", client.id, f"NAME={client.name} AUTO_CREATED=True")
                print(f"Auto-created missing client '{client.id}' ({client.name}).")
            else:
                print(f"Error: Client '{client_id}' not found.", file=sys.stderr)
                print("You can pass --create-client to auto-create it, or run:", file=sys.stderr)
                print(f"  python -m pdf_qa_agent.scripts.admin_cli clients create --name '{client_id}' --id {client_id}", file=sys.stderr)
                sys.exit(1)
        # Check if raw_key was explicitly specified
        raw_key_arg = getattr(args, "raw_key", None)
        if raw_key_arg and raw_key_arg.strip():
            raw_key = raw_key_arg.strip()
            import hashlib
            key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            key_prefix = raw_key[:14]
            existing_k = db.query(models.ApiKey).filter(models.ApiKey.key_hash == key_hash).first()
            if existing_k:
                print(f"Notice: API Key with prefix '{key_prefix}...' is already registered for client '{existing_k.client_id}'.", file=sys.stderr)
                if not existing_k.is_active:
                    existing_k.is_active = True
                    db.commit()
                    print(f"Re-enabled existing key (ID: {existing_k.id}).")
                return
        else:
            raw_key, key_hash, key_prefix = generate_api_key(prefix=getattr(args, "prefix", None) or "qscan_live_")

        key_record = models.ApiKey(
            client_id=client.id,
            key_hash=key_hash,
            prefix=key_prefix,
            name=args.name or f"Key for {client.name}",
            is_active=True
        )
        db.add(key_record)
        db.commit()
        db.refresh(key_record)

        log_audit("key_create", key_record.id, f"CLIENT_ID={client.id} PREFIX={key_prefix}")
        print("\n" + "=" * 70)
        print("API Key Successfully Created")
        print("=" * 70)
        print(f"Key ID:       {key_record.id}")
        print(f"Client ID:    {client.id} ({client.name})")
        print(f"Key Prefix:   {key_prefix}...")
        print(f"RAW API KEY:  {raw_key}")
        print("=" * 70)
        print("CRITICAL: Copy this raw key now. It is hashed with SHA-256 and will NEVER be shown again.")
        print(f"Header usage:  X-API-Key: {raw_key}")
        print("=" * 70 + "\n")
    finally:
        db.close()


def handle_keys_disable(args):
    db = database.SessionLocal()
    try:
        key = db.query(models.ApiKey).filter(models.ApiKey.id == args.key_id).first()
        if not key:
            print(f"Error: API key '{args.key_id}' not found.", file=sys.stderr)
            sys.exit(1)

        if not key.is_active:
            print(f"API key '{key.id}' is already inactive.")
            return

        if not confirm_prompt(f"Are you sure you want to disable API key '{key.id}' ({key.prefix}...)?", args.yes):
            print("Operation aborted by user.")
            sys.exit(0)

        key.is_active = False
        db.commit()
        log_audit("key_disable", key.id, f"CLIENT_ID={key.client_id}")
        print(f"API key '{key.id}' disabled successfully.")
    finally:
        db.close()


def handle_keys_enable(args):
    db = database.SessionLocal()
    try:
        key = db.query(models.ApiKey).filter(models.ApiKey.id == args.key_id).first()
        if not key:
            print(f"Error: API key '{args.key_id}' not found.", file=sys.stderr)
            sys.exit(1)

        if key.is_active:
            print(f"API key '{key.id}' is already active.")
            return

        if not confirm_prompt(f"Are you sure you want to enable API key '{key.id}' ({key.prefix}...)?", args.yes):
            print("Operation aborted by user.")
            sys.exit(0)

        key.is_active = True
        db.commit()
        log_audit("key_enable", key.id, f"CLIENT_ID={key.client_id}")
        print(f"API key '{key.id}' enabled successfully.")
    finally:
        db.close()


def handle_keys_delete(args):
    db = database.SessionLocal()
    try:
        key_id = getattr(args, "key_id", None) or getattr(args, "flag_key_id", None)
        client_id = getattr(args, "client_id", None)
        name = getattr(args, "name", None)

        if not key_id and not client_id:
            print("Error: Must specify key_id, client_id, or --client-id to delete keys.", file=sys.stderr)
            sys.exit(1)

        # Smart fallback: If key_id is provided, check if it is an ApiKey.id or a Client ID/name
        if key_id and not client_id:
            key_match = db.query(models.ApiKey).filter(models.ApiKey.id == key_id).first()
            if not key_match:
                client_match = db.query(models.Client).filter((models.Client.id == key_id) | (models.Client.name == key_id)).first()
                if client_match:
                    client_id = client_match.id
                    key_id = None

        query = db.query(models.ApiKey)
        if key_id:
            query = query.filter(models.ApiKey.id == key_id)
        if client_id:
            client = db.query(models.Client).filter((models.Client.id == client_id) | (models.Client.name == client_id)).first()
            if not client:
                print(f"Error: Client '{client_id}' not found.", file=sys.stderr)
                sys.exit(1)
            query = query.filter(models.ApiKey.client_id == client.id)
        if name:
            query = query.filter(models.ApiKey.name == name)

        keys = query.all()
        if not keys:
            print("No matching API keys found to delete.")
            return

        for key in keys:
            if not confirm_prompt(f"PERMANENT ACTION: Delete API key '{key.id}' ('{key.name}', {key.prefix}...)?", args.yes):
                print(f"Skipping key '{key.id}'.")
                continue

            k_id = key.id
            c_id = key.client_id
            db.delete(key)
            db.commit()

            log_audit("key_delete", k_id, f"CLIENT_ID={c_id}")
            print(f"API key '{k_id}' permanently deleted.")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="admin_cli",
        description="PDF QA Agent Admin CLI — Multi-tenant Clients and Keys Management"
    )
    subparsers = parser.add_subparsers(dest="entity", required=True)

    # 1. Clients Parser
    clients_parser = subparsers.add_parser("clients", help="Client management")
    client_sub = clients_parser.add_subparsers(dest="action", required=True)

    # clients list
    client_sub.add_parser("list", help="List all registered clients")

    # clients create
    p_create = client_sub.add_parser("create", help="Create a new client")
    p_create.add_argument("name", nargs="?", default=None, help="Human-readable name of the client")
    p_create.add_argument("--name", dest="flag_name", help="Human-readable name of the client (alias)")
    p_create.add_argument("--id", help="Optional custom Client ID (e.g. frontend1)")
    p_create.add_argument("--limit", "--daily-limit", dest="limit", type=int, default=None, help="Daily upload limit (default: None for Unlimited)")
    p_create.add_argument("--admin", action="store_true", help="Grant administrative privileges")

    # clients disable
    p_disable = client_sub.add_parser("disable", help="Disable a client")
    p_disable.add_argument("client_id", help="Client ID to disable")
    p_disable.add_argument("--force", action="store_true", help="Force disable even for master admin")
    p_disable.add_argument("-y", "--yes", action="store_true", help="Bypass confirmation prompt")

    # clients enable
    p_enable = client_sub.add_parser("enable", help="Enable a client")
    p_enable.add_argument("client_id", help="Client ID to enable")
    p_enable.add_argument("--force", action="store_true", help="Force enable")
    p_enable.add_argument("-y", "--yes", action="store_true", help="Bypass confirmation prompt")

    # clients limit
    p_limit = client_sub.add_parser("limit", help="Update client daily upload limit")
    p_limit.add_argument("client_id", help="Client ID to update")
    p_limit.add_argument("limit", type=int, help="New daily upload limit (use 0 for Unlimited)")
    p_limit.add_argument("--force", action="store_true", help="Force modification for internal clients")
    p_limit.add_argument("-y", "--yes", action="store_true", help="Bypass confirmation prompt")

    # clients delete
    p_cdelete = client_sub.add_parser("delete", help="Permanently delete a client and all associated data")
    p_cdelete.add_argument("client_id", help="Client ID to delete")
    p_cdelete.add_argument("--force", action="store_true", help="Force delete even for master admin")
    p_cdelete.add_argument("-y", "--yes", action="store_true", help="Bypass confirmation prompt")

    # 2. Keys Parser
    keys_parser = subparsers.add_parser("keys", help="API Key management")
    key_sub = keys_parser.add_subparsers(dest="action", required=True)

    # keys list
    p_klist = key_sub.add_parser("list", help="List API keys")
    p_klist.add_argument("--client_id", "--client-id", dest="client_id", help="Filter keys by client ID")

    # keys create
    p_kcreate = key_sub.add_parser("create", help="Generate or register an API key for a client")
    p_kcreate.add_argument("client_id", nargs="?", default=None, help="Client ID owning the key")
    p_kcreate.add_argument("--client-id", "--client_id", dest="flag_client_id", help="Client ID owning the key (alias)")
    p_kcreate.add_argument("--key", "--raw-key", dest="raw_key", help="Custom raw key string to add (instead of generating random)")
    p_kcreate.add_argument("--create-client", action="store_true", help="Auto-create the client if it does not exist")
    p_kcreate.add_argument("--name", help="Descriptive name for this key")
    p_kcreate.add_argument("--prefix", default="qscan_live_", help="Key prefix (default: 'qscan_live_')")

    # keys add (dedicated alias for adding specific keys)
    p_kadd = key_sub.add_parser("add", help="Add an existing API key for a client")
    p_kadd.add_argument("client_id", help="Client ID owning the key")
    p_kadd.add_argument("raw_key", help="The raw API key to add")
    p_kadd.add_argument("--name", help="Descriptive name for this key")
    p_kadd.add_argument("--create-client", action="store_true", help="Auto-create the client if it does not exist")

    # keys disable
    p_kdisable = key_sub.add_parser("disable", help="Revoke/disable an API key")
    p_kdisable.add_argument("key_id", help="Key ID to disable")
    p_kdisable.add_argument("-y", "--yes", action="store_true", help="Bypass confirmation prompt")

    # keys enable
    p_kenable = key_sub.add_parser("enable", help="Re-enable an API key")
    p_kenable.add_argument("key_id", help="Key ID to enable")
    p_kenable.add_argument("-y", "--yes", action="store_true", help="Bypass confirmation prompt")

    # keys delete
    p_kdelete = key_sub.add_parser("delete", help="Permanently delete an API key")
    p_kdelete.add_argument("key_id", nargs="?", default=None, help="Key ID to delete")
    p_kdelete.add_argument("--key-id", "--id", dest="flag_key_id", help="Key ID to delete (alias)")
    p_kdelete.add_argument("--client-id", "--client_id", dest="client_id", help="Delete key(s) belonging to this client")
    p_kdelete.add_argument("--name", help="Filter deletion by key name")
    p_kdelete.add_argument("-y", "--yes", action="store_true", help="Bypass confirmation prompt")

    args = parser.parse_args()

    dispatch = {
        ("clients", "list"): handle_clients_list,
        ("clients", "create"): handle_clients_create,
        ("clients", "disable"): handle_clients_disable,
        ("clients", "enable"): handle_clients_enable,
        ("clients", "limit"): handle_clients_limit,
        ("clients", "delete"): handle_clients_delete,
        ("keys", "list"): handle_keys_list,
        ("keys", "create"): handle_keys_create,
        ("keys", "add"): handle_keys_create,
        ("keys", "disable"): handle_keys_disable,
        ("keys", "enable"): handle_keys_enable,
        ("keys", "delete"): handle_keys_delete,
    }

    handler = dispatch.get((args.entity, args.action))
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
