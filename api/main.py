from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from .routes import router as pdf_router
from .font_routes import router as font_router
from .v1_routes import router as v1_router
from .auth import get_current_client, verify_internal_network
from .cleanup import run_startup_cleanup
from ..storage.database import engine, Base, init_db
from sqlalchemy import text

# Initialize database and migrate schema
init_db()

app = FastAPI(
    title="PDF QA Agent API",
    version="1.0.0",
    description="Multi-tenant PDF Question-Answer Scanner API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Primary V1 Routers
# Authenticated endpoints (strictly requires X-API-Key)
app.include_router(v1_router)

# 2. Legacy Routers (Strictly authenticated with X-API-Key; keyless requests return 401; network-restricted if enabled)
app.include_router(
    pdf_router,
    dependencies=[Depends(verify_internal_network), Depends(get_current_client)],
    tags=["legacy_pdfs"]
)
app.include_router(
    font_router,
    prefix="/api",
    dependencies=[Depends(verify_internal_network), Depends(get_current_client)],
    tags=["legacy_fonts"]
)
app.include_router(
    font_router,
    dependencies=[Depends(verify_internal_network), Depends(get_current_client)],
    tags=["legacy_fonts"]
)


@app.on_event("startup")
def on_startup():
    run_startup_cleanup()


# Health Check (strictly requires auth)
@app.get("/", dependencies=[Depends(get_current_client)], tags=["Health"])
def root():
    return {
        "service": "PDF QA Agent API",
        "status": "healthy",
        "version": "1.0.0"
    }
