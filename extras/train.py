import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import PruningNet
from utils import calculate_sparsity
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try different lambda values
lambdas = [1e-5, 1e-4, 1e-3]

transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

criterion = nn.CrossEntropyLoss()

def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)
            loss += gates.sum()
    return loss

def test_model(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total


results = []

# Train for each lambda
for lambda_sparse in lambdas:
    print(f"\nTraining with lambda = {lambda_sparse}")

    model = PruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
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

    # Evaluation
    accuracy = test_model(model)
    sparsity = calculate_sparsity(model)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")
    # Save model after training
    torch.save(model.state_dict(), f"model_lambda_{lambda_sparse}.pth")

    results.append((lambda_sparse, accuracy, sparsity))

# Print Final Table
print("\nFinal Results:")
print("Lambda\tAccuracy\tSparsity")
for r in results:
    print(f"{r[0]}\t{r[1]:.2f}%\t\t{r[2]:.2f}%")


# Plot Gate Distribution 
all_gates = []

for module in model.modules():
    if hasattr(module, "gate_scores"):
        gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
        all_gates.extend(gates.flatten())

plt.hist(all_gates, bins=50)
plt.title("Gate Value Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.show()