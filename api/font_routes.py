"""
API Routes for Font Inspection, Installation, and Decompilation.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import os

from ..ingest.font_installer import list_installed_fonts, install_reference_font, get_known_font_families
from .auth import get_current_client, get_current_admin_client
from ..storage import models

router = APIRouter(prefix="/fonts", tags=["Fonts"])


@router.get("/")
def get_installed_fonts(
    client: models.Client = Depends(get_current_client)
):
    """Returns a list of all installed font maps, reference fonts, and recognized font families."""
    fonts = list_installed_fonts()
    families = get_known_font_families()
    return {
        "installed_count": len(fonts),
        "known_families_count": len(families),
        "known_families": families,
        "fonts": fonts
    }


@router.post("/upload")
async def upload_font_file(
    file: UploadFile = File(...),
    admin_client: models.Client = Depends(get_current_admin_client)
):
    """
    Upload a TrueType (.ttf) or OpenType (.otf) font file.
    The system decompiles its glyph outlines, extracts GSUB ligatures,
    generates contour hashes, and installs it permanently into the system database.
    """
    if not file.filename.lower().endswith((".ttf", ".otf")):
        raise HTTPException(status_code=400, detail="Only .ttf and .otf font files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = install_reference_font(file_bytes, file.filename)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Installation failed."))

    return result
