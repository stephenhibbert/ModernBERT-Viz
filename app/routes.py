"""API routes for ModernBERT analysis."""

import logging
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from .model_ops import model_manager
from .analysis import create_token_trajectories, prepare_attention_data, validate_token_indices
from .config import AnalysisRequest, AnalysisResponse, ModelArchitecture

logger = logging.getLogger(__name__)

async def analyze_sentence(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze a sentence and return token trajectories and attention patterns."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Extract embeddings and attention
        layer_embeddings, layer_attention, tokens = model_manager.extract_layer_embeddings_and_attention(request.sentence)

        # Validate token indices
        valid_token_indices = validate_token_indices(request.token_indices, tokens)

        # Create trajectories
        trajectories, pca_variance = create_token_trajectories(
            layer_embeddings, tokens, valid_token_indices
        )

        # Prepare attention data
        attention_data = prepare_attention_data(layer_attention, tokens)

        # Get model architecture
        model_arch_data, total_params = model_manager.get_model_architecture()
        model_arch = [ModelArchitecture(**arch) for arch in model_arch_data]

        return AnalysisResponse(
            tokens=tokens,
            trajectories=trajectories,
            attention_data=attention_data,
            pca_explained_variance=pca_variance,
            model_architecture=model_arch,
            total_parameters=total_params
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def get_index() -> HTMLResponse:
    """Serve the main HTML page."""
    try:
        with open("templates/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Template file not found")

async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model_manager.is_loaded}