"""Main FastAPI application for ModernBERT Token Analysis."""

import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.model_ops import model_manager
from app.routes import analyze_sentence, get_index, health_check
from app.config import Settings, AnalysisRequest, AnalysisResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title=Settings.APP_TITLE, version=Settings.APP_VERSION)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer on startup."""
    await model_manager.initialize()

# API Routes
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(request: AnalysisRequest):
    """Analyze a sentence and return token trajectories and attention patterns."""
    return await analyze_sentence(request)

@app.get("/")
async def index():
    """Serve the main HTML page."""
    return await get_index()

@app.get("/health")
async def health():
    """Health check endpoint."""
    return await health_check()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)