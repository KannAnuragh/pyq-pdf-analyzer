import argparse
from ..storage import database
from ..api.auth import create_client_and_key


def main():
    parser = argparse.ArgumentParser(description="Create a client and generate an API key for PDF QA Agent API")
    parser.add_argument("--name", required=True, help="Name of the client (e.g. 'Portal PHP Server')")
    parser.add_argument("--limit", type=int, default=100, help="Daily upload limit (default: 100)")
    parser.add_argument("--prefix", default="qscan_live_", help="Prefix for key (default: 'qscan_live_')")
    args = parser.parse_args()

    db = database.SessionLocal()
    try:
        client, raw_key = create_client_and_key(
            db=db,
            name=args.name,
            daily_upload_limit=args.limit,
            prefix=args.prefix
        )
        print("=" * 60)
        print("Client successfully created!")
        print(f"Client ID:           {client.id}")
        print(f"Client Name:         {client.name}")
        print(f"Daily Upload Limit:  {client.daily_upload_limit}")
        print(f"API Key:             {raw_key}")
        print("=" * 60)
        print("IMPORTANT: Share this API key with the PHP client. It is hashed in the database and cannot be retrieved again.")
        print("Pass it as an HTTP header: 'X-API-Key: " + raw_key + "'")
        print("=" * 60)
    finally:
        db.close()


if __name__ == "__main__":
    main()
