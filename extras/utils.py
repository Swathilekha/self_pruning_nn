# utils.py

import torch

def calculate_sparsity(model, threshold=1e-2):
    total = 0
    zero = 0
    
    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            zero += (gates < threshold).sum().item()
    
    return 100 * zero / total