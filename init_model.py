import torch

from src.modeling_mimi import MimiModel
from src.configuration_mimi import MimiConfig

# Initializing a "kyutai/mimi" style configuration
configuration = MimiConfig()

# Initializing a model (with random weights) from the "kyutai/mimi" style configuration
model = MimiModel(configuration).to("cuda")

audio, quantized = model(torch.rand(1, 1, 48_000, device="cuda"), target_embeddings=torch.rand(1, 100, 1024, device="cuda"))

def count_model_parameters(model):
    """
    Counts the number of parameters in a model and prints them in a human-readable format.
    
    Args:
        model: The PyTorch model whose parameters are to be counted.
    
    Returns:
        A string representing the number of parameters in a human-readable format.
    """
    # Count the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Human-readable formatting
    for unit in ['parameters', 'K', 'M', 'B', 'T']:
        if total_params < 1000:
            return f"{total_params:.2f} {unit}"
        total_params /= 1000

    return f"{total_params:.2f} T"

print(count_model_parameters(model))

print(audio.shape)

for x in quantized:
    if isinstance(x, torch.Tensor):
        print(x.shape)

model.push_to_hub("AshwinSankar/Mimi-v1-multilingual", private=True)
# print("::::: Saving model to ./logs/first_checkpoint :::::")
# model.save_pretrained("logs/first_checkpoint")

# Accessing the model configuration
# configuration = model.config