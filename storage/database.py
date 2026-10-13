from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=FULL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """Create all tables that don't yet exist (idempotent)."""
    # Import models so their table definitions are registered on Base.metadata
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    
    # Safely migrate existing jobs table to add processing_time_seconds if missing
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN processing_time_seconds REAL"))
            conn.commit()
    except Exception:
        pass # Column already exists or table freshly created


# Auto-create tables on first import
init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
