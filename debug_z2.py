"""
Debug: Check Z_2 training
"""

import math
import torch
import torch.nn as nn
import numpy as np


class ZkBundle(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.input_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
        self.output_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
    
    def forward(self, x1, x2):
        p1 = self.input_phases[x1]
        p2 = self.input_phases[x2]
        phi = (p1 + p2) % (2 * math.pi)
        dists = torch.abs(phi.unsqueeze(-1) - self.output_phases.unsqueeze(0))
        dists = dists % (2 * math.pi)
        dists = torch.min(dists, 2 * math.pi - dists)
        return -dists
    
    def get_phases(self):
        return self.input_phases.detach().cpu().numpy()


def train_zk(k, n_samples=1000, epochs=150, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model


# Test Z_2
print("=== Testing Z_2 ===")
model2 = train_zk(2, seed=0)

x1 = torch.tensor([0, 0, 1, 1])
x2 = torch.tensor([0, 1, 0, 1])
y_true = (x1 + x2) % 2

with torch.no_grad():
    out = model2(x1, x2)
    pred = out.argmax(1)
    print(f"Inputs: x1={x1.tolist()}, x2={x2.tolist()}")
    print(f"True:   {y_true.tolist()}")
    print(f"Pred:   {pred.tolist()}")
    print(f"Match:  {(pred == y_true).all().item()}")

print(f"\nLearned input_phases: {model2.get_phases()}")
print(f"Expected: [0, pi]")

# Check output distribution
print("\n=== Testing all Z_2 outputs ===")
x1 = torch.randint(0, 2, (100,))
x2 = torch.randint(0, 2, (100,))
y_true = (x1 + x2) % 2

with torch.no_grad():
    out = model2(x1, x2)
    pred = out.argmax(1)
    acc = (pred == y_true).float().mean().item()
    print(f"Accuracy: {acc:.2%}")

# Test Z_3
print("\n=== Testing Z_3 ===")
model3 = train_zk(3, seed=0)

x1 = torch.randint(0, 3, (100,))
x2 = torch.randint(0, 3, (100,))
y_true = (x1 + x2) % 3

with torch.no_grad():
    out = model3(x1, x2)
    pred = out.argmax(1)
    acc = (pred == y_true).float().mean().item()
    print(f"Accuracy: {acc:.2%}")
    print(f"Learned input_phases: {model3.get_phases()}")
    print(f"Expected: [0, 2pi/3, 4pi/3]")
