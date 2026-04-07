"""Main FastAPI application for PulseML dashboard."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import json

from config.settings import ROOT_DIR, MODEL_PATH, EDA_DIR
from src.api.schemas import HealthResponse
from src.api.routes import predict, report
from src.api.routes.predict import load_model

# Initialize FastAPI app
app = FastAPI(
    title="PulseML — Employee Attrition Predictor",
    description="ML-powered prediction of employee attrition risk with SHAP explanations.",
    version="1.0.0",
)

# Setup static file serving for charts
STATIC_DIR = ROOT_DIR / "reports"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include routers
app.include_router(predict.router)
app.include_router(report.router)


@app.on_event("startup")
async def startup():
    """Load model on application startup."""
    try:
        load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Model failed to load on startup: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and model status."""
    from src.api.routes.predict import MODEL
    
    model_loaded = MODEL is not None
    model_name = type(MODEL).__name__ if MODEL else "None"
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_name=model_name,
        version="1.0.0",
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Serve main dashboard."""
    dashboard_file = ROOT_DIR / "templates" / "dashboard.html"
    if not dashboard_file.exists():
        return "<h1>Dashboard not found</h1>"
    
    return dashboard_file.read_text(encoding='utf-8')


@app.get("/report", response_class=HTMLResponse)
async def report_page() -> str:
    """Serve EDA report with charts."""
    report_file = ROOT_DIR / "templates" / "report.html"
    if not report_file.exists():
        return "<h1>Report not found</h1>"
    
    return report_file.read_text(encoding='utf-8')


@app.get("/api/stats")
async def get_stats() -> dict:
    """Get overall dataset statistics."""
    return {
        "dataset_size": 1470,
        "attrition_rate": 0.161,
        "at_risk_count": 237,
        "model_auc": 0.797,
        "features_used": 25,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
