"""
Experiment 2: Final CRT Characterization
=========================================
FINDING: Network implements CRT ONLY when:
  1. gcd(k1, k2) = 1
  2. min(k1, k2) >= 3

Z_2 is too small to encode the group homomorphism properly.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import json


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


def generate_zk_data(k, n_samples):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    return x1, x2, y


def train_zk(k, n_samples=1000, epochs=150, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1, x2, y = generate_zk_data(k, n_samples)
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model


def test_crt_composition(model_a, model_b, m_a, m_b, m_target, n_samples=500):
    x1 = torch.randint(0, m_target, (n_samples,))
    x2 = torch.randint(0, m_target, (n_samples,))
    y_true = (x1 + x2) % m_target
    
    x1_a = x1 % m_a
    x2_a = x2 % m_a
    x1_b = x1 % m_b
    x2_b = x2 % m_b
    
    y_a_true = (x1_a + x2_a) % m_a
    y_b_true = (x1_b + x2_b) % m_b
    
    with torch.no_grad():
        out_a = model_a(x1_a, x2_a)
        out_b = model_b(x1_b, x2_b)
        
        pred_a = out_a.argmax(1)
        pred_b = out_b.argmax(1)
        
        acc_a = (pred_a == y_a_true).float().mean().item()
        acc_b = (pred_b == y_b_true).float().mean().item()
        
        closures = 0
        for i in range(n_samples):
            a = pred_a[i].item()
            b = pred_b[i].item()
            
            combined = crt_reconstruct(a, b, m_a, m_b)
            
            if combined is not None and combined == y_true[i].item():
                closures += 1
        
        closure_rate = closures / n_samples
    
    return {
        'acc_a': acc_a,
        'acc_b': acc_b,
        'closure_rate': closure_rate,
        'closures': closures,
        'total': n_samples
    }


def main():
    print("="*60)
    print("EXPERIMENT 2: FINAL CRT CHARACTERIZATION")
    print("="*60)
    print("\nHYPOTHESIS: CRT requires min(k1,k2) >= 3")
    print("(Z_2 is too small to encode homomorphism)\n")
    
    compositions = [
        {'name': 'Z_3 x Z_5 -> Z_15', 'k1': 3, 'k2': 5, 'k_target': 15, 'min_k': 3},
        {'name': 'Z_5 x Z_7 -> Z_35', 'k1': 5, 'k2': 7, 'k_target': 35, 'min_k': 5},
        {'name': 'Z_2 x Z_3 -> Z_6',  'k1': 2, 'k2': 3, 'k_target': 6,  'min_k': 2},
        {'name': 'Z_2 x Z_5 -> Z_10', 'k1': 2, 'k2': 5, 'k_target': 10, 'min_k': 2},
        {'name': 'Z_2 x Z_7 -> Z_14', 'k1': 2, 'k2': 7, 'k_target': 14, 'min_k': 2},
        {'name': 'Z_3 x Z_7 -> Z_21', 'k1': 3, 'k2': 7, 'k_target': 21, 'min_k': 3},
    ]
    
    n_seeds = 15
    results = {}
    
    for comp in compositions:
        print(f"\n=== {comp['name']} (min_k={comp['min_k']}) ===")
        
        closure_rates = []
        
        for seed in range(n_seeds):
            model_a = train_zk(comp['k1'], n_samples=1000, seed=seed)
            model_b = train_zk(comp['k2'], n_samples=1000, seed=seed+100)
            
            result = test_crt_composition(
                model_a, model_b, 
                comp['k1'], comp['k2'], comp['k_target'],
                n_samples=500
            )
            
            closure_rates.append(result['closure_rate'])
        
        mean_closure = np.mean(closure_rates)
        
        results[comp['name']] = {
            'k1': comp['k1'],
            'k2': comp['k2'],
            'k_target': comp['k_target'],
            'min_k': comp['min_k'],
            'closure_rates': closure_rates,
            'mean_closure': mean_closure,
        }
        
        status = "SUCCEED" if mean_closure > 0.5 else "FAIL"
        print(f"    Closure: {mean_closure:.2%} [{status}]")
    
    print("\n" + "="*60)
    print("TABLE 1: CRT COMPOSITION MATRIX")
    print("="*60)
    print("\n| Composition       | min(k) | Closure  | Status |")
    print("|-------------------|--------|----------|--------|")
    
    for name, r in results.items():
        status = "OK" if (r['min_k'] >= 3) == (r['mean_closure'] > 0.5) else "ANOMALY"
        print(f"| {name:17s} | {r['min_k']:6d} | {r['mean_closure']:7.2%} | {status:6s} |")
    
    # Analysis
    min_k_2 = [r for r in results.values() if r['min_k'] < 3]
    min_k_3 = [r for r in results.values() if r['min_k'] >= 3]
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print(f"\nmin(k) >= 3:")
    print(f"  Success: {sum(1 for r in min_k_3 if r['mean_closure'] > 0.5)}/{len(min_k_3)}")
    print(f"  Mean closure: {np.mean([r['mean_closure'] for r in min_k_3]):.2%}")
    
    print(f"\nmin(k) < 3:")
    print(f"  Success: {sum(1 for r in min_k_2 if r['mean_closure'] > 0.5)}/{len(min_k_2)}")
    print(f"  Mean closure: {np.mean([r['mean_closure'] for r in min_k_2]):.2%}")
    
    with open('experiment_2_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\nSaved to experiment_2_results.json")
    
    return results


if __name__ == "__main__":
    main()
