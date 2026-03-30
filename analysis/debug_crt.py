"""
Debug: Verify Z_3 x Z_5 -> Z_15 still works
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


def extended_gcd(a, b):
    if b == 0:
        return (1, 0, a)
    else:
        x1, y1, g = extended_gcd(b, a % b)
        return (y1, x1 - (a // b) * y1, g)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def crt_reconstruct(a1, a2, m1, m2):
    if gcd(m1, m2) != 1:
        return None
    inv1 = extended_gcd(m1, m2)[0] % m2
    inv2 = extended_gcd(m2, m1)[0] % m1
    x = (a1 * m2 * inv1 + a2 * m1 * inv2) % (m1 * m2)
    return x


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


# Test Z_3 x Z_5 -> Z_15
print("=== Testing Z_3 x Z_5 -> Z_15 ===")

model3 = train_zk(3, seed=0)
model5 = train_zk(5, seed=1)

n_test = 500
x1 = torch.randint(0, 15, (n_test,))
x2 = torch.randint(0, 15, (n_test,))
y_true = (x1 + x2) % 15

x1_mod3 = x1 % 3
x2_mod3 = x2 % 3
x1_mod5 = x1 % 5
x2_mod5 = x2 % 5

y_mod3_true = (x1_mod3 + x2_mod3) % 3
y_mod5_true = (x1_mod5 + x2_mod5) % 5

with torch.no_grad():
    out3 = model3(x1_mod3, x2_mod3)
    out5 = model5(x1_mod5, x2_mod5)
    
    pred3 = out3.argmax(1)
    pred5 = out5.argmax(1)
    
    acc3 = (pred3 == y_mod3_true).float().mean().item()
    acc5 = (pred5 == y_mod5_true).float().mean().item()
    
    print(f"Z_3 accuracy: {acc3:.2%}")
    print(f"Z_5 accuracy: {acc5:.2%}")
    
    closures = 0
    for i in range(n_test):
        a3 = pred3[i].item()
        a5 = pred5[i].item()
        
        # This is the OLD method from experiment B
        # Just add the predictions and take mod 15
        combined_old = (a3 + a5) % 15
        
        # This is the NEW method using CRT
        combined_new = crt_reconstruct(a3, a5, 3, 5)
        
        if combined_old == y_true[i].item():
            closures += 1
    
    print(f"\nClosure rate (old method): {closures/n_test:.2%}")
    print(f"Expected: close to 100%")
