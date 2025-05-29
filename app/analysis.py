"""Analysis functions for token trajectories and attention processing."""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Tuple
from .config import TokenTrajectory, AttentionData

def create_token_trajectories(layer_embeddings: List[np.ndarray], tokens: List[str],
                              token_indices: List[int]) -> Tuple[List[TokenTrajectory], float]:
    """Create 2D trajectories for specified tokens using PCA."""
    if not token_indices:
        return [], 0.0

    # Extract embeddings for specified tokens across all layers
    token_embeddings_by_layer = []
    for layer_idx, embeddings in enumerate(layer_embeddings):
        layer_token_embeddings = []
        for token_idx in token_indices:
            if token_idx < len(embeddings):
                layer_token_embeddings.append(embeddings[token_idx])
        if layer_token_embeddings:
            token_embeddings_by_layer.append(np.array(layer_token_embeddings))

    if not token_embeddings_by_layer:
        return [], 0.0

    # Concatenate all embeddings for PCA fitting
    all_embeddings = np.vstack(token_embeddings_by_layer)

    # Fit PCA
    pca = PCA(n_components=2, random_state=42)
    all_projected = pca.fit_transform(all_embeddings)

    # Split back into trajectories for each token
    trajectories = []
    n_tokens = len(token_indices)
    n_layers = len(token_embeddings_by_layer)

    for token_pos, token_idx in enumerate(token_indices):
        trajectory_points = []
        for layer_idx in range(n_layers):
            point_idx = layer_idx * n_tokens + token_pos
            if point_idx < len(all_projected):
                trajectory_points.append(all_projected[point_idx].tolist())

        trajectories.append(TokenTrajectory(
            token=tokens[token_idx] if token_idx < len(tokens) else f"Token_{token_idx}",
            token_index=token_idx,
            trajectory=trajectory_points
        ))

    explained_variance = float(np.sum(pca.explained_variance_ratio_))
    return trajectories, explained_variance

def prepare_attention_data(layer_attention: List[np.ndarray], tokens: List[str]) -> List[AttentionData]:
    """Prepare attention data for visualization (sample every 3rd layer to reduce data size)."""
    attention_data = []
    for layer_idx in range(0, len(layer_attention), 3):
        attention_data.append(AttentionData(
            layer=layer_idx,
            attention_matrix=layer_attention[layer_idx].tolist(),
            tokens=tokens
        ))
    return attention_data

def validate_token_indices(token_indices: List[int], tokens: List[str], max_tokens: int = 6) -> List[int]:
    """Validate and limit token indices."""
    valid_token_indices = [idx for idx in token_indices if 0 <= idx < len(tokens)]
    
    if len(valid_token_indices) > max_tokens:
        valid_token_indices = valid_token_indices[:max_tokens]
    
    return valid_token_indices