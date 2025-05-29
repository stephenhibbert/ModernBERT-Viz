# app.py - FastAPI Backend for ModernBERT Token Analysis
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, ModernBertModel
import json
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ModernBERT Token Analysis", version="1.0.0")

# Global model and tokenizer
tokenizer = None
model = None


class AnalysisRequest(BaseModel):
    sentence: str
    token_indices: List[int] = []  # Up to 6 token indices to compare


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
    details: dict


class AnalysisResponse(BaseModel):
    tokens: List[str]
    trajectories: List[TokenTrajectory]
    attention_data: List[AttentionData]
    pca_explained_variance: float
    model_architecture: List[ModelArchitecture]
    total_parameters: int


@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer on startup"""
    global tokenizer, model

    logger.info("Loading ModernBERT model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        # Force eager attention implementation to avoid the warning
        model = ModernBertModel.from_pretrained(
            "answerdotai/ModernBERT-base",
            attn_implementation="eager"
        )
        model.eval()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


def extract_layer_embeddings_and_attention(text: str):
    """Extract embeddings and attention from all layers"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    # Get hidden states and attention weights
    hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
    attention_weights = outputs.attentions  # Tuple of (batch_size, num_heads, seq_len, seq_len)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Convert to numpy and extract embeddings
    layer_embeddings = []
    for layer_output in hidden_states:
        embeddings = layer_output[0].numpy()  # Remove batch dimension
        layer_embeddings.append(embeddings)

    # Convert attention weights to numpy and average across heads
    layer_attention = []
    for layer_attn in attention_weights:
        # Average across attention heads: (num_heads, seq_len, seq_len) -> (seq_len, seq_len)
        avg_attention = np.mean(layer_attn[0].numpy(), axis=0)
        layer_attention.append(avg_attention)

    return layer_embeddings, layer_attention, tokens


def get_model_architecture():
    """Extract model architecture information dynamically"""
    architecture = []

    def count_parameters(module):
        return sum(p.numel() for p in module.parameters())

    def count_direct_parameters(module):
        """Count only parameters directly owned by this module (not submodules)"""
        return sum(p.numel() for p in module.parameters(recurse=False))

    def get_layer_details(name, module):
        details = {
            'class': module.__class__.__name__,
            'parameters': count_direct_parameters(module)
        }

        # Add specific details based on layer type
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            details['in_features'] = module.in_features
            details['out_features'] = module.out_features
            details['bias'] = module.bias is not None
        elif hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'):
            details['vocab_size'] = module.num_embeddings
            details['embedding_dim'] = module.embedding_dim
        elif hasattr(module, 'normalized_shape'):
            details['normalized_shape'] = list(module.normalized_shape)
            details['eps'] = module.eps
        elif hasattr(module, 'p') and hasattr(module, 'inplace'):
            details['dropout_prob'] = module.p

        return details

    # Get total parameters for the entire model
    total_params = count_parameters(model)

    # Embeddings
    embeddings = model.embeddings
    embeddings_total_params = count_parameters(embeddings)
    arch_entry = ModelArchitecture(
        layer_name="embeddings",
        layer_type="ModernBertEmbeddings",
        parameters=embeddings_total_params,
        details=get_layer_details("embeddings", embeddings)
    )
    architecture.append(arch_entry)

    # Add embedding sub-components
    for name, module in embeddings.named_children():
        sub_params = count_parameters(module)
        sub_arch = ModelArchitecture(
            layer_name=f"embeddings.{name}",
            layer_type=module.__class__.__name__,
            parameters=sub_params,
            details=get_layer_details(f"embeddings.{name}", module)
        )
        architecture.append(sub_arch)

    # Encoder layers - calculate total for all layers first
    all_layers_params = count_parameters(model.layers)
    single_layer_params = count_parameters(model.layers[0])

    # Add summary entry for all layers
    arch_entry = ModelArchitecture(
        layer_name="layers",
        layer_type=f"ModuleList (22x ModernBertEncoderLayer)",
        parameters=all_layers_params,
        details={'class': 'ModuleList', 'num_layers': 22, 'params_per_layer': single_layer_params}
    )
    architecture.append(arch_entry)

    # Show detailed breakdown for first layer as example
    layer_0 = model.layers[0]
    layer_0_params = count_parameters(layer_0)
    arch_entry = ModelArchitecture(
        layer_name="layers.0",
        layer_type="ModernBertEncoderLayer",
        parameters=layer_0_params,
        details=get_layer_details("layers.0", layer_0)
    )
    architecture.append(arch_entry)

    # Add sub-components of layer 0
    for name, module in layer_0.named_children():
        sub_params = count_parameters(module)
        sub_arch = ModelArchitecture(
            layer_name=f"layers.0.{name}",
            layer_type=module.__class__.__name__,
            parameters=sub_params,
            details=get_layer_details(f"layers.0.{name}", module)
        )
        architecture.append(sub_arch)

        # Add attention sub-components if it's the attention module
        if name == "attn":
            for attn_name, attn_module in module.named_children():
                attn_params = count_parameters(attn_module)
                attn_arch = ModelArchitecture(
                    layer_name=f"layers.0.{name}.{attn_name}",
                    layer_type=attn_module.__class__.__name__,
                    parameters=attn_params,
                    details=get_layer_details(f"layers.0.{name}.{attn_name}", attn_module)
                )
                architecture.append(attn_arch)

        # Add MLP sub-components if it's the MLP module
        elif name == "mlp":
            for mlp_name, mlp_module in module.named_children():
                mlp_params = count_parameters(mlp_module)
                mlp_arch = ModelArchitecture(
                    layer_name=f"layers.0.{name}.{mlp_name}",
                    layer_type=mlp_module.__class__.__name__,
                    parameters=mlp_params,
                    details=get_layer_details(f"layers.0.{name}.{mlp_name}", mlp_module)
                )
                architecture.append(mlp_arch)

    # Final norm
    final_norm_params = count_parameters(model.final_norm)
    arch_entry = ModelArchitecture(
        layer_name="final_norm",
        layer_type="LayerNorm",
        parameters=final_norm_params,
        details=get_layer_details("final_norm", model.final_norm)
    )
    architecture.append(arch_entry)

    return architecture, total_params


def create_token_trajectories(layer_embeddings: List[np.ndarray], tokens: List[str],
                              token_indices: List[int]) -> tuple:
    """Create 2D trajectories for specified tokens using PCA"""
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
    """Create 2D trajectories for specified tokens using PCA"""
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


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_sentence(request: AnalysisRequest):
    """Analyze a sentence and return token trajectories and attention patterns"""
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Extract embeddings and attention
        layer_embeddings, layer_attention, tokens = extract_layer_embeddings_and_attention(request.sentence)

        # Validate token indices
        valid_token_indices = [idx for idx in request.token_indices
                               if 0 <= idx < len(tokens)]

        if len(valid_token_indices) > 6:
            valid_token_indices = valid_token_indices[:6]

        # Create trajectories
        trajectories, pca_variance = create_token_trajectories(
            layer_embeddings, tokens, valid_token_indices
        )

        # Prepare attention data (sample every 3rd layer to reduce data size)
        attention_data = []
        for layer_idx in range(0, len(layer_attention), 3):
            attention_data.append(AttentionData(
                layer=layer_idx,
                attention_matrix=layer_attention[layer_idx].tolist(),
                tokens=tokens
            ))

        # Get model architecture
        model_arch, total_params = get_model_architecture()

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


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    return HTMLResponse(content=HTML_CONTENT)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


# HTML Content
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModernBERT Token Analysis</title>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
            overflow-x: hidden;
        }

        .app-container {
            display: flex;
            min-height: 100vh;
        }

        .main-content {
            flex: 1;
            display: flex;
            gap: 20px;
            padding: 20px;
            overflow: hidden;
        }

        .analysis-panel {
            flex: 2;
            overflow-y: auto;
            padding-right: 10px;
        }

        .architecture-panel {
            flex: 1;
            min-width: 350px;
            max-width: 400px;
            background: #2c3e50;
            color: white;
            padding: 20px;
            overflow-y: auto;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            height: fit-content;
            max-height: calc(100vh - 40px);
            position: sticky;
            top: 20px;
        }

        .architecture-header {
            text-align: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #34495e;
            padding-bottom: 15px;
        }

        .architecture-header h2 {
            margin: 0;
            color: #ecf0f1;
            font-size: 1.4em;
        }

        .architecture-stats {
            font-size: 0.9em;
            color: #bdc3c7;
            margin-top: 8px;
        }

        .architecture-tree {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.3;
        }

        .arch-item {
            margin: 1px 0;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .arch-item:hover {
            background-color: #34495e;
            transform: translateX(2px);
        }

        .arch-item.highlighted {
            background-color: #3498db;
            color: white;
            box-shadow: 0 0 10px rgba(52, 152, 219, 0.5);
        }

        .arch-item.attention-layer {
            background-color: #e74c3c;
            color: white;
        }

        .arch-item .layer-info {
            display: flex;
            flex-direction: column;
            flex: 1;
        }

        .arch-item .layer-name {
            font-weight: bold;
            font-size: 12px;
        }

        .arch-item .layer-type {
            color: #95a5a6;
            font-size: 10px;
            margin-top: 1px;
        }

        .arch-item .param-count {
            color: #7f8c8d;
            font-size: 11px;
            font-weight: bold;
            min-width: 50px;
            text-align: right;
        }

        .arch-item.highlighted .param-count,
        .arch-item.attention-layer .param-count {
            color: white;
        }

        .indent-1 { margin-left: 12px; }
        .indent-2 { margin-left: 24px; }
        .indent-3 { margin-left: 36px; }

        .toggle-architecture {
            display: none;
        }

        @media (max-width: 1200px) {
            .main-content {
                flex-direction: column;
            }

            .architecture-panel {
                position: relative;
                top: 0;
                max-height: 300px;
                min-width: auto;
                max-width: none;
                margin-top: 20px;
            }
        }

        @media (max-width: 768px) {
            .main-content {
                padding: 10px;
                gap: 10px;
            }

            .toggle-architecture {
                display: block;
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
                background: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
            }

            .architecture-panel {
                display: none;
                position: fixed;
                top: 60px;
                right: 20px;
                bottom: 20px;
                left: 20px;
                z-index: 999;
                max-height: none;
            }

            .architecture-panel.mobile-visible {
                display: block;
            }
        }

        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
            height: fit-content;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
        }

        .header h1 {
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }

        .header p {
            color: #7f8c8d;
            font-size: 1.1em;
            margin: 10px 0 0 0;
        }

        .input-section {
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }

        input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ced4da;
            border-radius: 6px;
            font-size: 16px;
            transition: border-color 0.3s;
            box-sizing: border-box;
        }

        input[type="text"]:focus {
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
        }

        .token-selector {
            display: none;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
            max-height: 200px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            background: #f8f9fa;
        }

        .token-chip {
            padding: 8px 12px;
            border: 2px solid #dee2e6;
            border-radius: 20px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
            user-select: none;
            font-size: 14px;
            position: relative;
            min-width: 60px;
            text-align: center;
        }

        .token-chip:hover {
            border-color: #007bff;
            background-color: #f8f9fa;
            transform: translateY(-1px);
        }

        .token-chip.selected {
            background-color: #007bff;
            color: white;
            border-color: #007bff;
        }

        .token-chip.selected::after {
            content: attr(data-order);
            position: absolute;
            top: -8px;
            right: -8px;
            background: #dc3545;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }

        .selected-count {
            font-size: 12px;
            color: #6c757d;
            margin-top: 8px;
        }

        button {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
        }

        button:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results {
            display: none;
        }

        .plot-container {
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.1);
        }

        .plot-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #2c3e50;
            text-align: center;
        }

        .attention-controls {
            text-align: center;
            margin: 20px 0;
        }

        .layer-selector {
            margin: 0 10px;
            padding: 8px 16px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }

        .error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #f5c6cb;
            margin: 20px 0;
        }

        .info-box {
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #bee5eb;
            margin: 20px 0;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="main-panel">
            <div class="container">
                <div class="header">
                    <h1>🧠 ModernBERT Token Analysis</h1>
                    <p>Explore how token embeddings evolve through layers and attention patterns</p>
                </div>

        <div class="input-section">
            <div class="form-group">
                <label for="sentence-input">Enter a sentence to analyze:</label>
                <input type="text" id="sentence-input" placeholder="The cat sat on the mat." 
                       value="The cat sat on the mat.">
            </div>

            <div class="form-group">
                <label>Select up to 6 tokens to compare trajectories:</label>
                <div id="token-selector" class="token-selector"></div>
                <div id="selected-count" class="selected-count">Select tokens after tokenizing the sentence</div>
            </div>

            <button id="tokenize-btn" onclick="tokenizeSentence()">Tokenize Sentence</button>
            <button id="analyze-btn" onclick="analyzeSentence()" disabled>Analyze Tokens</button>
        </div>

        <div id="loading" class="loading" style="display: none;">
            <div class="spinner"></div>
            <p>Analyzing sentence through ModernBERT layers...</p>
        </div>

        <div id="error" class="error" style="display: none;"></div>

        <div id="results" class="results">
            <div class="info-box">
                <strong>Token Trajectories:</strong> Shows how selected tokens move through the embedding space across layers using PCA projection. Select up to 6 tokens to compare their paths.<br>
                <strong>Attention Heatmaps:</strong> Displays attention patterns showing which tokens attend to which others at different layers.
                <br><br>
                <strong>Tip:</strong> Try selecting different types of tokens (content words, function words, punctuation) to see how they behave differently through the layers.
            </div>

            <div class="plot-container">
                <div class="plot-title">Token Trajectories Through Layers</div>
                <div id="trajectory-plot"></div>
                <div style="text-align: center; margin-top: 10px; color: #6c757d; font-size: 14px;">
                    <span id="pca-variance"></span>
                </div>
            </div>

            <div class="plot-container">
                <div class="plot-title">Attention Patterns</div>
                <div class="attention-controls">
                    <label for="layer-select">Layer:</label>
                    <select id="layer-select" class="layer-selector" onchange="updateAttentionPlot()">
                    </select>
                </div>
                <div id="attention-plot">                </div>
            </div>
        </div>

        <div class="architecture-panel">
            <div class="architecture-header">
                <h2>🏗️ Model Architecture</h2>
                <div class="architecture-stats">
                    <div id="total-params">Loading...</div>
                    <div>ModernBERT-base</div>
                </div>
            </div>
            <div id="architecture-tree" class="architecture-tree">
                Loading architecture...
            </div>
        </div>
    </div>

    <button class="toggle-architecture" onclick="toggleArchitecture()">
        Toggle Architecture
    </button>

    <script>
        let currentTokens = [];
        let selectedTokens = [];
        let analysisData = null;
        let currentHighlightedLayer = null;

        async function tokenizeSentence() {
            const sentence = document.getElementById('sentence-input').value.trim();
            if (!sentence) {
                showError('Please enter a sentence');
                return;
            }

            showLoading(true);
            hideError();

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        sentence: sentence,
                        token_indices: []
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                currentTokens = data.tokens;
                selectedTokens = [];

                displayTokenSelector();
                document.getElementById('analyze-btn').disabled = true;

            } catch (error) {
                showError('Failed to tokenize sentence: ' + error.message);
            } finally {
                showLoading(false);
            }
        }

        function displayTokenSelector() {
            const selector = document.getElementById('token-selector');
            selector.innerHTML = '';
            selector.style.display = 'flex';

            currentTokens.forEach((token, index) => {
                const chip = document.createElement('div');
                chip.className = 'token-chip';
                chip.textContent = `${index}: ${token}`;
                chip.onclick = () => toggleToken(index, chip);
                selector.appendChild(chip);
            });

            updateSelectedCount();
        }

        function toggleToken(index, chipElement) {
            if (selectedTokens.includes(index)) {
                selectedTokens = selectedTokens.filter(i => i !== index);
                chipElement.classList.remove('selected');
                chipElement.removeAttribute('data-order');
            } else if (selectedTokens.length < 6) {
                selectedTokens.push(index);
                chipElement.classList.add('selected');
                chipElement.setAttribute('data-order', selectedTokens.length);
            } else {
                // Show a brief message that max tokens reached
                showMessage('Maximum 6 tokens can be selected', 'warning');
            }

            updateSelectedCount();
            document.getElementById('analyze-btn').disabled = selectedTokens.length === 0;
        }

        function updateSelectedCount() {
            const countEl = document.getElementById('selected-count');
            if (selectedTokens.length === 0) {
                countEl.textContent = 'No tokens selected (select 1-6 tokens)';
                countEl.style.color = '#6c757d';
            } else {
                const tokenNames = selectedTokens.map(i => currentTokens[i]).join(', ');
                countEl.textContent = `Selected: ${tokenNames} (${selectedTokens.length}/6)`;
                countEl.style.color = selectedTokens.length >= 4 ? '#e74c3c' : '#28a745';
            }
        }

        async function analyzeSentence() {
            if (selectedTokens.length === 0) {
                showError('Please select at least one token');
                return;
            }

            const sentence = document.getElementById('sentence-input').value.trim();
            showLoading(true);
            hideError();

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        sentence: sentence,
                        token_indices: selectedTokens
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                analysisData = await response.json();
                displayResults();
                displayModelArchitecture();

            } catch (error) {
                showError('Analysis failed: ' + error.message);
            } finally {
                showLoading(false);
            }
        }

        function displayModelArchitecture() {
            const treeContainer = document.getElementById('architecture-tree');
            const totalParamsEl = document.getElementById('total-params');

            if (!analysisData || !analysisData.model_architecture) {
                return;
            }

            // Update total parameters
            const totalParams = analysisData.total_parameters;
            totalParamsEl.textContent = `${(totalParams / 1000000).toFixed(1)}M parameters`;

            const grouped = {};
            analysisData.model_architecture.forEach(item => {
                const parts = item.layer_name.split('.');
                const mainComponent = parts[0];

                if (!grouped[mainComponent]) {
                    grouped[mainComponent] = [];
                }
                grouped[mainComponent].push(item);
            });

            let html = '';

            // Embeddings
            if (grouped.embeddings) {
                html += createArchitectureSection('embeddings', grouped.embeddings);
            }

            // Encoder layers - show summary then detailed example
            if (grouped.layers) {
                const layersSummary = grouped.layers.find(item => item.layer_name === 'layers');
                if (layersSummary) {
                    html += `
                        <div class="arch-item" style="margin-top: 10px;">
                            <span class="layer-name">layers</span>
                            <span class="layer-type">(22x EncoderLayer)</span>
                            <span class="param-count">${formatParamCount(layersSummary.parameters)}</span>
                        </div>
                    `;
                }

                // Show detailed breakdown for layer 0
                const layer0Items = grouped.layers.filter(item => 
                    item.layer_name.startsWith('layers.0') && item.layer_name !== 'layers'
                );
                html += createLayerDetailedSection(layer0Items);

                // Add note about other layers
                html += '<div class="arch-item indent-1" style="color: #7f8c8d; font-style: italic;">... (layers 1-21 have identical structure) ...</div>';
            }

            // Final norm
            if (grouped.final_norm) {
                html += createArchitectureSection('final_norm', grouped.final_norm);
            }

            treeContainer.innerHTML = html;
        }

        function createArchitectureSection(name, items) {
            let html = '';

            items.forEach(item => {
                const indent = item.layer_name.split('.').length - 1;
                const indentClass = `indent-${Math.min(indent, 3)}`;
                const paramCount = formatParamCount(item.parameters);

                let displayName = item.layer_name.split('.').pop();
                if (displayName === 'tok_embeddings') displayName = 'tok_embeddings';
                else if (displayName.includes('norm')) displayName = displayName;
                else if (displayName === 'drop') displayName = 'drop';

                html += `
                    <div class="arch-item ${indentClass}" data-layer="${item.layer_name}">
                        <div class="layer-info">
                            <span class="layer-name">${displayName}</span>
                            <span class="layer-type">${item.layer_type}</span>
                        </div>
                        <span class="param-count">${paramCount}</span>
                    </div>
                `;
            });

            return html;
        }

        function createLayerDetailedSection(items) {
            let html = '';

            // Find the main layer item
            const mainLayer = items.find(item => item.layer_name === 'layers.0');
            if (mainLayer) {
                html += `
                    <div class="arch-item indent-1" data-layer="layers.0">
                        <span class="layer-name">[0] EncoderLayer (example)</span>
                        <span class="param-count">${formatParamCount(mainLayer.parameters)}</span>
                    </div>
                `;
            }

            // Group sub-components
            const components = {};
            items.forEach(item => {
                if (item.layer_name === 'layers.0') return; // Skip main layer

                const parts = item.layer_name.split('.');
                if (parts.length === 3) {
                    // Direct components like attn, mlp, norms
                    const compName = parts[2];
                    if (!components[compName]) components[compName] = [];
                    components[compName].push(item);
                } else if (parts.length === 4) {
                    // Sub-components like attn.Wqkv, mlp.Wi
                    const parentComp = parts[2];
                    if (!components[parentComp]) components[parentComp] = [];
                    components[parentComp].push(item);
                }
            });

            // Display components in order
            const componentOrder = ['attn_norm', 'attn', 'mlp_norm', 'mlp'];
            componentOrder.forEach(compName => {
                if (components[compName]) {
                    components[compName].forEach(item => {
                        const parts = item.layer_name.split('.');
                        const isSubComponent = parts.length === 4;
                        const indentClass = isSubComponent ? 'indent-3' : 'indent-2';

                        let displayName = parts[parts.length - 1];
                        if (compName === 'attn' && !isSubComponent) {
                            displayName = 'attention';
                        } else if (compName === 'mlp' && !isSubComponent) {
                            displayName = 'mlp';
                        } else if (displayName.includes('norm')) {
                            displayName = displayName;
                        } else if (displayName === 'Wqkv') {
                            displayName = 'Wqkv (Q,K,V projection)';
                        } else if (displayName === 'Wo') {
                            displayName = 'Wo (output projection)';
                        } else if (displayName === 'Wi') {
                            displayName = 'Wi (input projection)';
                        } else if (displayName === 'rotary_emb') {
                            displayName = 'rotary_emb';
                        }

                        const isAttention = compName === 'attn' && !isSubComponent;
                        const attentionClass = isAttention ? 'attention-layer' : '';

                        html += `
                            <div class="arch-item ${indentClass} ${attentionClass}" data-layer="${item.layer_name}">
                                <div class="layer-info">
                                    <span class="layer-name">${displayName}</span>
                                </div>
                                <span class="param-count">${formatParamCount(item.parameters)}</span>
                            </div>
                        `;
                    });
                }
            });

            return html;
        }

        function formatParamCount(count) {
            if (count >= 1000000) {
                return `${(count / 1000000).toFixed(1)}M`;
            } else if (count >= 1000) {
                return `${(count / 1000).toFixed(1)}K`;
            } else {
                return count.toString();
            }
        }

        function highlightArchitectureLayer(layerIndex) {
            // Remove previous highlights
            document.querySelectorAll('.arch-item.highlighted').forEach(el => {
                el.classList.remove('highlighted');
            });

            // Highlight current layer
            const layerElement = document.querySelector(`[data-layer="layers.${layerIndex}"]`);
            if (layerElement) {
                layerElement.classList.add('highlighted');
                layerElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }

            currentHighlightedLayer = layerIndex;
        }

        function showMessage(message, type = 'info') {
            // Create a temporary message element
            const messageEl = document.createElement('div');
            messageEl.className = `message ${type}`;
            messageEl.textContent = message;
            messageEl.style.cssText = `
                position: fixed; 
                top: 20px; 
                right: 20px; 
                padding: 12px 20px; 
                border-radius: 6px; 
                z-index: 1000;
                font-weight: 500;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transform: translateX(100%);
                transition: transform 0.3s ease;
                ${type === 'warning' ? 'background: #fff3cd; color: #856404; border: 1px solid #ffeaa7;' : 'background: #d4edda; color: #155724; border: 1px solid #c3e6cb;'}
            `;

            document.body.appendChild(messageEl);

            // Animate in
            setTimeout(() => {
                messageEl.style.transform = 'translateX(0)';
            }, 100);

            // Remove after 3 seconds
            setTimeout(() => {
                messageEl.style.transform = 'translateX(100%)';
                setTimeout(() => {
                    if (messageEl.parentNode) {
                        messageEl.parentNode.removeChild(messageEl);
                    }
                }, 300);
            }, 3000);
        }

        function displayResults() {
            document.getElementById('results').style.display = 'block';

            // Display trajectory plot
            plotTrajectories();

            // Setup attention plot
            setupAttentionControls();
            updateAttentionPlot();
        }

        function plotTrajectories() {
            const colorPalette = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
            ];

            const traces = analysisData.trajectories.map((traj, index) => {
                const color = colorPalette[index % colorPalette.length];

                const x = traj.trajectory.map(point => point[0]);
                const y = traj.trajectory.map(point => point[1]);

                // Create layer annotations for trajectory points
                const layerLabels = x.map((_, layerIdx) => `Layer ${layerIdx}`);

                return {
                    x: x,
                    y: y,
                    mode: 'lines+markers',
                    name: `${traj.token} (${traj.token_index})`,
                    line: { 
                        color: color, 
                        width: 3,
                        shape: 'spline',
                        smoothing: 0.3
                    },
                    marker: { 
                        size: 8, 
                        color: color,
                        line: { color: 'white', width: 2 },
                        symbol: index % 2 === 0 ? 'circle' : 'square'
                    },
                    text: layerLabels,
                    hovertemplate: `<b>${traj.token}</b> (idx: ${traj.token_index})<br>%{text}<br>PCA1: %{x:.3f}<br>PCA2: %{y:.3f}<extra></extra>`
                };
            });

            // Add start and end annotations for each trajectory
            const annotations = [];
            analysisData.trajectories.forEach((traj, index) => {
                const color = colorPalette[index % colorPalette.length];
                const x = traj.trajectory.map(point => point[0]);
                const y = traj.trajectory.map(point => point[1]);

                if (x.length > 0) {
                    // Start point
                    annotations.push({
                        x: x[0],
                        y: y[0],
                        text: 'START',
                        showarrow: true,
                        arrowhead: 2,
                        arrowsize: 1,
                        arrowwidth: 2,
                        arrowcolor: color,
                        font: { size: 10, color: color },
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: color,
                        borderwidth: 1
                    });

                    // End point
                    annotations.push({
                        x: x[x.length - 1],
                        y: y[y.length - 1],
                        text: 'END',
                        showarrow: true,
                        arrowhead: 2,
                        arrowsize: 1,
                        arrowwidth: 2,
                        arrowcolor: color,
                        font: { size: 10, color: color },
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: color,
                        borderwidth: 1
                    });
                }
            });

            const layout = {
                xaxis: { 
                    title: 'PCA Component 1',
                    gridcolor: '#e0e0e0',
                    zerolinecolor: '#d0d0d0'
                },
                yaxis: { 
                    title: 'PCA Component 2',
                    gridcolor: '#e0e0e0',
                    zerolinecolor: '#d0d0d0'
                },
                hovermode: 'closest',
                showlegend: true,
                legend: { 
                    x: 0, 
                    y: 1,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#d0d0d0',
                    borderwidth: 1
                },
                margin: { t: 30, b: 50, l: 60, r: 30 },
                plot_bgcolor: '#fafafa',
                annotations: annotations
            };

            Plotly.newPlot('trajectory-plot', traces, layout, {responsive: true});

            // Update PCA variance info
            const variancePercent = (analysisData.pca_explained_variance * 100).toFixed(1);
            document.getElementById('pca-variance').innerHTML = 
                `PCA explains <strong>${variancePercent}%</strong> of variance | Showing <strong>${analysisData.trajectories.length}</strong> token trajectories across <strong>${analysisData.trajectories[0]?.trajectory.length || 0}</strong> layers`;
        }

        function setupAttentionControls() {
            const layerSelect = document.getElementById('layer-select');
            layerSelect.innerHTML = '';

            analysisData.attention_data.forEach(data => {
                const option = document.createElement('option');
                option.value = data.layer;
                option.textContent = `Layer ${data.layer}`;
                layerSelect.appendChild(option);
            });
        }

        function updateAttentionPlot() {
            const selectedLayer = parseInt(document.getElementById('layer-select').value);
            const layerData = analysisData.attention_data.find(d => d.layer === selectedLayer);

            if (!layerData) return;

            // Highlight the corresponding layer in architecture
            highlightArchitectureLayer(selectedLayer);

            const trace = {
                z: layerData.attention_matrix,
                x: layerData.tokens,
                y: layerData.tokens,
                type: 'heatmap',
                colorscale: 'Viridis',
                hovertemplate: 'From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>'
            };

            const layout = {
                xaxis: { 
                    title: 'To Token',
                    tickangle: 45,
                    side: 'bottom'
                },
                yaxis: { 
                    title: 'From Token',
                    autorange: 'reversed'
                },
                margin: { t: 30, b: 100, l: 100, r: 30 }
            };

            Plotly.newPlot('attention-plot', [trace], layout, {responsive: true});
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }

        function showError(message) {
            const errorEl = document.getElementById('error');
            errorEl.textContent = message;
            errorEl.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error').style.display = 'none';
        }

        // Initialize by tokenizing the default sentence
        window.onload = function() {
            tokenizeSentence();
        };
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)