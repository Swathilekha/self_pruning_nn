"""
Self-Pruning Neural Network
---------------------------
Implements a neural network that learns to prune itself using
learnable gate parameters and L1 sparsity regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 1. Prunable Linear Layer
# ============================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Standard weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable gate scores
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Convert gate scores to [0,1]
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gating
        pruned_weights = self.weight * gates
        
        return F.linear(x, pruned_weights, self.bias)


# ============================
# 2. Model Definition (CNN + Prunable Layers)
# ============================
class PruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Prunable fully connected layers
        self.fc1 = PrunableLinear(32 * 8 * 8, 128)
        self.fc2 = PrunableLinear(128, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================
# 3. Sparsity Loss (L1 on gates)
# ============================
def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)
            loss += gates.sum()
    return loss


# ============================
# 4. Test Accuracy Function
# ============================
def test_model(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total


# ============================
# 5. Sparsity Calculation
# ============================
def calculate_sparsity(model, threshold=1e-2):
    total = 0
    zero = 0

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            zero += (gates < threshold).sum().item()

    return 100 * zero / total


# ============================
# 6. Data Loading
# ============================
transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)


# ============================
# 7. Training Loop (Multi-Lambda)
# ============================
lambdas = [1e-5, 1e-4, 1e-3]
criterion = nn.CrossEntropyLoss()

results = []
best_model = None
best_accuracy = 0

for lambda_sparse in lambdas:
    print(f"\nTraining with lambda = {lambda_sparse}")

    model = PruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(10):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            s_loss = sparsity_loss(model)
            total = loss + lambda_sparse * s_loss

            total.backward()
            optimizer.step()

            total_loss += total.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Evaluate
    accuracy = test_model(model, test_loader)
    sparsity = calculate_sparsity(model)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    results.append((lambda_sparse, accuracy, sparsity))

    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model


# ============================
# 8. Print Results Table
# ============================
print("\nFinal Results:")
print("Lambda\tAccuracy\tSparsity")
for r in results:
    print(f"{r[0]}\t{r[1]:.2f}%\t\t{r[2]:.2f}%")


# ============================
# 9. Compression Ratio
# ============================
total_params = 0
active_params = 0

for module in best_model.modules():
    if hasattr(module, "gate_scores"):
        gates = torch.sigmoid(module.gate_scores)
        total_params += gates.numel()
        active_params += (gates >= 1e-2).sum().item()

compression = total_params / active_params
print(f"\nCompression Ratio: {compression:.2f}x")


# ============================
# 10. Gate Distribution Plot
# ============================
import numpy as np

all_gates = []

for module in best_model.modules():
    if hasattr(module, "gate_scores"):
        gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
        all_gates.extend(gates.flatten())

all_gates = np.array(all_gates)

# Split for better visualization
small_vals = all_gates[all_gates < 0.1]
large_vals = all_gates[all_gates >= 0.1]

plt.figure()
plt.hist(small_vals, bins=50, alpha=0.7, label="Pruned (near 0)")
plt.hist(large_vals, bins=50, alpha=0.7, label="Active")

plt.title("Gate Value Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.legend()

plt.savefig("gate_distribution.png")
plt.show()
