# Self-Pruning Neural Network (CIFAR-10)

This project implements a **self-pruning neural network** that dynamically removes unimportant weights during training using a learnable gating mechanism.

---

## Objective

To design a neural network that:
- Learns which weights are unnecessary
- Automatically prunes them during training
- Maintains performance while reducing model complexity

---

## Approach

- Implemented a custom **PrunableLinear layer**
- Each weight has a **learnable gate parameter**
- Gates are passed through a **sigmoid function (0–1)**
- Applied **L1 regularization** on gates to encourage sparsity

### Loss Function

Total Loss = Classification Loss + λ × Sparsity Loss

---

## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 1e-5   | 66.21%   | 34.61%   |
| 1e-4   | 63.43%   | 42.29%   |
| 1e-3   | 61.23%   | 51.68%   |

---

## Compression

- **Compression Ratio:** 1.53x  
- Indicates reduction in active parameters while maintaining performance

---

## Key Insights

- Increasing λ leads to higher sparsity
- Accuracy decreases gradually, showing a **controlled trade-off**
- Even at ~51% sparsity, the model retains **~61% accuracy**
- Confirms that neural networks contain **redundant parameters**

---

## Gate Distribution

- Majority of gate values are pushed towards **0**
- Small fraction remains active
- Demonstrates effective pruning behavior

---

## Architecture

- CNN-based feature extractor (for CIFAR-10 images)
- Fully connected layers replaced with **PrunableLinear layers**

---

## Tech Stack

- Python
- PyTorch
- CIFAR-10 Dataset
- Matplotlib

---

## Project Structure

self-pruning-nn/
│
├── model.py # Prunable layer + network
├── train.py # Training + evaluation
├── utils.py # Sparsity calculation
├── compression.py # Compression ratio
├── report.md # Case study report
└── README.md


---

## How to Run

pip install torch torchvision matplotlib
python train.py
python compression.py

## Conclusion

This project demonstrates how neural networks can adaptively prune themselves, reducing complexity while maintaining reasonable accuracy. It highlights the effectiveness of L1 regularization in inducing sparsity and the presence of redundancy in deep learning models.

## Author

Swathilekha V
