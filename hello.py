from transformers import AutoModel

model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

# Print model summary
print(model)

# Get parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")