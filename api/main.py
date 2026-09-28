from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router as pdf_router
from .font_routes import router as font_router
from ..storage.database import engine, Base
from sqlalchemy import text

# Create DB tables
Base.metadata.create_all(bind=engine)

# Ensure file_path column exists for legacy databases
try:
    with engine.connect() as conn:
        res = conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
        col_names = [r[1] for r in res]
        if "file_path" not in col_names and len(col_names) > 0:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN file_path VARCHAR"))
            conn.commit()
except Exception as e:
    print(f"Database schema check warning: {e}")

app = FastAPI(title="PDF QA Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdf_router, tags=["pdfs"])
app.include_router(font_router, prefix="/api")
app.include_router(font_router)

from .routes import trigger_queue_worker

@app.on_event("startup")
def on_startup():
    trigger_queue_worker()

@app.get("/")
def root():
    return {"message": "Welcome to PDF QA Agent API"}
