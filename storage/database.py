from sqlalchemy import create_engine
try:
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError:
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", os.path.join(DATA_DIR, "uploads"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

DEFAULT_DB_PATH = os.path.join(DATA_DIR, "pdf_qa.db")
SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

if SQLALCHEMY_DATABASE_URL.startswith("sqlite:///"):
    db_path = SQLALCHEMY_DATABASE_URL.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

from sqlalchemy import event

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30}
)

if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=DELETE")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """Create all tables that don't yet exist (idempotent)."""
    # Import models so their table definitions are registered on Base.metadata
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    
    # Safely migrate existing jobs table to add new columns if missing
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            res = conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
            col_names = [r[1] for r in res]
            if "processing_time_seconds" not in col_names and len(col_names) > 0:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN processing_time_seconds REAL"))
            if "file_path" not in col_names and len(col_names) > 0:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN file_path VARCHAR"))
            if "client_id" not in col_names and len(col_names) > 0:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN client_id VARCHAR"))
            if "stage" not in col_names and len(col_names) > 0:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN stage VARCHAR DEFAULT 'pending'"))
            if "error_message" not in col_names and len(col_names) > 0:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN error_message TEXT"))
            if "updated_at" not in col_names and len(col_names) > 0:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN updated_at DATETIME"))
            
            # Migrate clients table for is_admin
            c_res = conn.execute(text("PRAGMA table_info(clients)")).fetchall()
            c_cols = [r[1] for r in c_res]
            if "is_admin" not in c_cols and len(c_cols) > 0:
                conn.execute(text("ALTER TABLE clients ADD COLUMN is_admin BOOLEAN DEFAULT 0"))

            # Migrate api_keys table for last_used_at
            k_res = conn.execute(text("PRAGMA table_info(api_keys)")).fetchall()
            k_cols = [r[1] for r in k_res]
            if "last_used_at" not in k_cols and len(k_cols) > 0:
                conn.execute(text("ALTER TABLE api_keys ADD COLUMN last_used_at DATETIME"))

            conn.commit()
    except Exception as e:
        print(f"Database schema migration notice: {e}")

    # Dedicated Admin Client (id: c_admin)
    # Master key configured via ADMIN_API_KEY in environment or deployment
    admin_key = os.getenv("ADMIN_API_KEY", "qscan_live_admin_master_key_2026").strip()

    try:
        import hashlib
        with SessionLocal() as db:
            admin_client = db.query(models.Client).filter(models.Client.id == "c_admin").first()
            if not admin_client:
                admin_client = models.Client(
                    id="c_admin",
                    name="System Administrator",
                    daily_upload_limit=None,
                    is_active=True,
                    is_admin=True
                )
                db.add(admin_client)
                db.commit()
                db.refresh(admin_client)
            else:
                admin_client.is_admin = True
                admin_client.is_active = True
                admin_client.daily_upload_limit = None
                db.commit()

            # Provision key for Admin
            akh = hashlib.sha256(admin_key.encode("utf-8")).hexdigest()
            ak_rec = db.query(models.ApiKey).filter(models.ApiKey.key_hash == akh).first()
            if not ak_rec:
                db.add(models.ApiKey(
                    client_id=admin_client.id,
                    key_hash=akh,
                    prefix=admin_key[:14],
                    name="Master Admin API Key",
                    is_active=True
                ))
            else:
                ak_rec.client_id = admin_client.id
                ak_rec.is_active = True
            db.commit()

            # Provision declared client keys from environment or deployment
            # Format: CLIENT_API_KEYS="client1:key1,client2:key2"
            declared_keys = os.getenv("CLIENT_API_KEYS", "").strip()
            if declared_keys:
                for pair in declared_keys.split(","):
                    pair = pair.strip()
                    if ":" in pair:
                        cid, raw_k = pair.split(":", 1)
                        cid = cid.strip()
                        raw_k = raw_k.strip()
                        if cid and raw_k:
                            c = db.query(models.Client).filter(models.Client.id == cid).first()
                            if not c:
                                c = models.Client(id=cid, name=f"Client {cid}", daily_upload_limit=None, is_active=True)
                                db.add(c)
                                db.commit()
                                db.refresh(c)
                            k_hash = hashlib.sha256(raw_k.encode("utf-8")).hexdigest()
                            k_rec = db.query(models.ApiKey).filter(models.ApiKey.key_hash == k_hash).first()
                            if not k_rec:
                                db.add(models.ApiKey(
                                    client_id=c.id,
                                    key_hash=k_hash,
                                    prefix=raw_k[:14],
                                    name=f"Configured Key for {cid}",
                                    is_active=True
                                ))
                                db.commit()
    except Exception as e:
        print(f"Notice: Admin client auto-provisioning: {e}")


# Auto-create tables on first import
init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
