"""
Debug: Check if network discovers group structure or just memorizes
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


def extended_gcd(a, b):
    if b == 0:
        return (1, 0, a)
    else:
        x1, y1, g = extended_gcd(b, a % b)
        return (y1, x1 - (a // b) * y1, g)


def crt_reconstruct(a1, a2, m1, m2):
    while np.gcd(m1, m2) != 1:
        return None
    inv1 = extended_gcd(m1, m2)[0] % m2
    inv2 = extended_gcd(m2, m1)[0] % m1
    x = (a1 * m2 * inv1 + a2 * m1 * inv2) % (m1 * m2)
    return x


def train_zk(k, n_samples=1000, epochs=200, seed=42):
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


# Test different scenarios
print("=== Testing if network discovers group structure ===\n")

# Scenario 1: Train on random Z_3 problems, test on structured inputs
print("Scenario 1: Random training, test on structured")
model3 = train_zk(3, seed=0)

# Generate structured test: x1 always 0, x2 varies
x1 = torch.zeros(30, dtype=torch.long)
x2 = torch.arange(30) % 3
y_true = (x1 + x2) % 3

with torch.no_grad():
    out = model3(x1, x2)
    pred = out.argmax(1)
    print(f"  Input (0, 0..2) -> True: {y_true.tolist()}")
    print(f"  Input (0, 0..2) -> Pred: {pred.tolist()}")
    print(f"  Match: {(pred == y_true).all().item()}")

# Check learned phases
print(f"\n  Learned input_phases: {model3.get_phases()}")
print(f"  Expected: [0, 2π/3, 4π/3] ≈ [0, 2.09, 4.19]")

# Now test CRT with more seeds
print("\n=== Testing Z_3 x Z_5 -> Z_15 with 20 seeds ===")
closure_rates = []
for seed in range(20):
    model3 = train_zk(3, seed=seed)
    model5 = train_zk(5, seed=seed+100)
    
    n_test = 100
    x1 = torch.randint(0, 15, (n_test,))
    x2 = torch.randint(0, 15, (n_test,))
    y_true = (x1 + x2) % 15
    
    x1_mod3 = x1 % 3
    x2_mod3 = x2 % 3
    x1_mod5 = x1 % 5
    x2_mod5 = x2 % 5
    
    with torch.no_grad():
        out3 = model3(x1_mod3, x2_mod3)
        out5 = model5(x1_mod5, x2_mod5)
        pred3 = out3.argmax(1)
        pred5 = out5.argmax(1)
        
        # Test: does the network output satisfy the CRT equation?
        # For a Z_3 + Z_5 problem, we need (pred3 + pred5) % 15 to equal y_true
        # But the network doesn't know about the target modulus!
        
        closures = 0
        for i in range(n_test):
            # Check if predictions satisfy the group structure
            # Combined prediction should be: (pred3 + pred5) % 15
            combined = (pred3[i].item() + pred5[i].item()) % 15
            if combined == y_true[i].item():
                closures += 1
        
        closure_rates.append(closures / n_test)

print(f"  Mean closure rate: {np.mean(closure_rates):.2%}")
print(f"  This is what we got with simple addition, NOT CRT")
