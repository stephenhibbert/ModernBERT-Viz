"""Model operations for ModernBERT token analysis."""

import logging
import torch
import numpy as np
from transformers import AutoTokenizer, ModernBertModel
from typing import Tuple, List

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ModernBERT model and tokenizer."""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._model_name = "answerdotai/ModernBERT-base"
    
    async def initialize(self):
        """Load model and tokenizer."""
        logger.info("Loading ModernBERT model and tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self.model = ModernBertModel.from_pretrained(
                self._model_name,
                attn_implementation="eager"
            )
            self.model.eval()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def extract_layer_embeddings_and_attention(self, text: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """Extract embeddings and attention from all layers."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
        
        hidden_states = outputs.hidden_states
        attention_weights = outputs.attentions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Convert to numpy and extract embeddings
        layer_embeddings = []
        for layer_output in hidden_states:
            embeddings = layer_output[0].numpy()
            layer_embeddings.append(embeddings)
        
        # Convert attention weights to numpy and average across heads
        layer_attention = []
        for layer_attn in attention_weights:
            avg_attention = np.mean(layer_attn[0].numpy(), axis=0)
            layer_attention.append(avg_attention)
        
        return layer_embeddings, layer_attention, tokens
    
    def get_model_architecture(self):
        """Extract model architecture information dynamically."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
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
        
        architecture = []
        total_params = count_parameters(self.model)
        
        # Embeddings
        embeddings = self.model.embeddings
        embeddings_total_params = count_parameters(embeddings)
        architecture.append({
            'layer_name': "embeddings",
            'layer_type': "ModernBertEmbeddings",
            'parameters': embeddings_total_params,
            'details': get_layer_details("embeddings", embeddings)
        })
        
        # Add embedding sub-components
        for name, module in embeddings.named_children():
            sub_params = count_parameters(module)
            architecture.append({
                'layer_name': f"embeddings.{name}",
                'layer_type': module.__class__.__name__,
                'parameters': sub_params,
                'details': get_layer_details(f"embeddings.{name}", module)
            })
        
        # Encoder layers
        all_layers_params = count_parameters(self.model.layers)
        single_layer_params = count_parameters(self.model.layers[0])
        
        architecture.append({
            'layer_name': "layers",
            'layer_type': f"ModuleList (22x ModernBertEncoderLayer)",
            'parameters': all_layers_params,
            'details': {'class': 'ModuleList', 'num_layers': 22, 'params_per_layer': single_layer_params}
        })
        
        # Show detailed breakdown for first layer as example
        layer_0 = self.model.layers[0]
        layer_0_params = count_parameters(layer_0)
        architecture.append({
            'layer_name': "layers.0",
            'layer_type': "ModernBertEncoderLayer",
            'parameters': layer_0_params,
            'details': get_layer_details("layers.0", layer_0)
        })
        
        # Add sub-components of layer 0
        for name, module in layer_0.named_children():
            sub_params = count_parameters(module)
            architecture.append({
                'layer_name': f"layers.0.{name}",
                'layer_type': module.__class__.__name__,
                'parameters': sub_params,
                'details': get_layer_details(f"layers.0.{name}", module)
            })
            
            # Add attention sub-components if it's the attention module
            if name == "attn":
                for attn_name, attn_module in module.named_children():
                    attn_params = count_parameters(attn_module)
                    architecture.append({
                        'layer_name': f"layers.0.{name}.{attn_name}",
                        'layer_type': attn_module.__class__.__name__,
                        'parameters': attn_params,
                        'details': get_layer_details(f"layers.0.{name}.{attn_name}", attn_module)
                    })
            
            # Add MLP sub-components if it's the MLP module
            elif name == "mlp":
                for mlp_name, mlp_module in module.named_children():
                    mlp_params = count_parameters(mlp_module)
                    architecture.append({
                        'layer_name': f"layers.0.{name}.{mlp_name}",
                        'layer_type': mlp_module.__class__.__name__,
                        'parameters': mlp_params,
                        'details': get_layer_details(f"layers.0.{name}.{mlp_name}", mlp_module)
                    })
        
        # Final norm
        final_norm_params = count_parameters(self.model.final_norm)
        architecture.append({
            'layer_name': "final_norm",
            'layer_type': "LayerNorm",
            'parameters': final_norm_params,
            'details': get_layer_details("final_norm", self.model.final_norm)
        })
        
        return architecture, total_params
    
    @property
    def is_loaded(self) -> bool:
        """Check if model and tokenizer are loaded."""
        return self.model is not None and self.tokenizer is not None

# Global model manager instance
model_manager = ModelManager()