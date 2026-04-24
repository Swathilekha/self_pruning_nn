import torch
import matplotlib.pyplot as plt
from model import PruningNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BEST model (use best accuracy one)
model = PruningNet().to(device)
model.load_state_dict(torch.load("model_lambda_1e-05.pth"))
model.eval()

# Collect gate values
all_gates = []

for module in model.modules():
    if hasattr(module, "gate_scores"):
        gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
        all_gates.extend(gates.flatten())

# Convert to tensor for easier filtering
import numpy as np
all_gates = np.array(all_gates)

# 🔥 Fix visualization (IMPORTANT)
small_vals = all_gates[all_gates < 0.1]   # focus near 0
large_vals = all_gates[all_gates >= 0.1]

plt.figure()

# Plot small values (zoomed)
plt.hist(small_vals, bins=50, alpha=0.7, label="Near 0 (Pruned)")

# Plot large values
plt.hist(large_vals, bins=50, alpha=0.7, label="Active Weights")

plt.title("Improved Gate Value Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.legend()

# Save for report
plt.savefig("gate_distribution.png")
plt.show()