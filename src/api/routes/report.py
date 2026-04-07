"""Report endpoint for viewing EDA charts."""

from fastapi import APIRouter
from pathlib import Path
from config.settings import EDA_DIR

router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/report-images")
async def get_eda_images() -> dict:
    """
    Get list of available EDA report images.
    
    Returns:
        Dict with image filenames and relative paths
    """
    if not EDA_DIR.exists():
        return {"images": [], "message": "EDA directory not found"}
    
    image_files = sorted(EDA_DIR.glob("*.png"))
    
    images = [
        {
            "filename": img.name,
            "path": f"/static/eda/{img.name}",
            "title": img.stem.replace("_", " ").title(),
        }
        for img in image_files
    ]
    
    return {
        "total": len(images),
        "images": images,
    }
