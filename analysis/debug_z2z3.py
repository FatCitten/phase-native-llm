"""
Debug: Check Z_2 x Z_3 -> Z_6 predictions
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


def crt_reconstruct(a1, a2, m1, m2):
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


# Test Z_2 x Z_3 -> Z_6
print("=== Testing Z_2 x Z_3 -> Z_6 ===")

model2 = train_zk(2, seed=0)
model3 = train_zk(3, seed=1)

n_test = 20
x1 = torch.randint(0, 6, (n_test,))
x2 = torch.randint(0, 6, (n_test,))
y_true = (x1 + x2) % 6

x1_mod2 = x1 % 2
x2_mod2 = x2 % 2
x1_mod3 = x1 % 3
x2_mod3 = x2 % 3

y_mod2_true = (x1_mod2 + x2_mod2) % 2
y_mod3_true = (x1_mod3 + x2_mod3) % 3

with torch.no_grad():
    out2 = model2(x1_mod2, x2_mod2)
    out3 = model3(x1_mod3, x2_mod3)
    
    pred2 = out2.argmax(1)
    pred3 = out3.argmax(1)
    
    print("x1 x2 | y_true | y_mod2 | y_mod3 | pred2 | pred3 | CRT | match")
    print("-" * 70)
    
    for i in range(n_test):
        crt = crt_reconstruct(pred2[i].item(), pred3[i].item(), 2, 3)
        match = "YES" if crt == y_true[i].item() else "NO"
        print(f" {x1[i]}  {x2[i]}  |   {y_true[i]}    |   {y_mod2_true[i]}    |   {y_mod3_true[i]}    |   {pred2[i]}    |   {pred3[i]}    |  {crt}  | {match}")
    
    # Check individual accuracies
    acc2 = (pred2 == y_mod2_true).float().mean().item()
    acc3 = (pred3 == y_mod3_true).float().mean().item()
    print(f"\nZ_2 accuracy: {acc2:.2%}")
    print(f"Z_3 accuracy: {acc3:.2%}")
