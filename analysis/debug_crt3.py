"""
Compare CRT implementations
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


# Original CRT (from experiment_b)
def crt_reconstruct_v1(a3, a5):
    m1, m2 = 3, 5
    inv1 = extended_gcd(m1, m2)[0] % m2
    inv2 = extended_gcd(m2, m1)[0] % m1
    x = (a3 * m2 * inv1 + a5 * m1 * inv2) % (m1 * m2)
    return x


# New CRT (from experiment_2)
def crt_reconstruct_v2(a1, a2, m1, m2):
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

n_test = 100
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
    
    # What are the actual predictions?
    print(f"pred3[:10]: {pred3[:10].tolist()}")
    print(f"pred5[:10]: {pred5[:10].tolist()}")
    print(f"y_true[:10]: {y_true[:10].tolist()}")
    
    # Test both CRT methods
    closures_v1 = 0
    closures_v2 = 0
    simple_add = 0
    
    for i in range(n_test):
        a3 = pred3[i].item()
        a5 = pred5[i].item()
        
        # Method 1: original (hardcoded 3,5)
        r1 = crt_reconstruct_v1(a3, a5)
        
        # Method 2: new (general)
        r2 = crt_reconstruct_v2(a3, a5, 3, 5)
        
        # Method 3: simple addition
        r3 = (a3 + a5) % 15
        
        if r1 == y_true[i].item():
            closures_v1 += 1
        if r2 == y_true[i].item():
            closures_v2 += 1
        if r3 == y_true[i].item():
            simple_add += 1
    
    print(f"\nMethod 1 (original CRT): {closures_v1/n_test:.2%}")
    print(f"Method 2 (new CRT): {closures_v2/n_test:.2%}")
    print(f"Method 3 (simple add): {simple_add/n_test:.2%}")
