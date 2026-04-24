import torch
from model import PruningNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best model (choose one)
model = PruningNet().to(device)
model.load_state_dict(torch.load("model_lambda_1e-05.pth"))  # best accuracy
model.eval()

total_params = 0
active_params = 0

for module in model.modules():
    if hasattr(module, "gate_scores"):
        gates = torch.sigmoid(module.gate_scores)
        total_params += gates.numel()
        active_params += (gates >= 1e-2).sum().item()

compression = total_params / active_params

print(f"Compression Ratio: {compression:.2f}x")