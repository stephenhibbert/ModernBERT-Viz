"""Configuration and data models for ModernBERT analysis."""

from pydantic import BaseModel
from typing import List, Dict, Any, Literal

class AnalysisRequest(BaseModel):
    sentence: str
    token_indices: List[int] = []  # Up to 6 token indices to compare
    reduction_method: Literal["pca", "tsne"] = "pca"  # Dimensionality reduction method

class TokenTrajectory(BaseModel):
    token: str
    token_index: int
    trajectory: List[List[float]]  # List of [x, y] coordinates for each layer

class AttentionData(BaseModel):
    layer: int
    attention_matrix: List[List[float]]
    tokens: List[str]

class ModelArchitecture(BaseModel):
    layer_name: str
    layer_type: str
    parameters: int
    details: Dict[str, Any]

class AnalysisResponse(BaseModel):
    tokens: List[str]
    trajectories: List[TokenTrajectory]
    attention_data: List[AttentionData]
    explained_variance: float  # Works for both PCA and t-SNE (KL divergence for t-SNE)
    reduction_method: str
    model_architecture: List[ModelArchitecture]
    total_parameters: int

# Application settings
class Settings:
    APP_TITLE = "ModernBERT Token Analysis"
    APP_VERSION = "1.0.0"
    MODEL_NAME = "answerdotai/ModernBERT-base"
    MAX_TOKENS_TO_COMPARE = 6
    ATTENTION_LAYER_SAMPLING = 3  # Sample every 3rd layer for attention visualization