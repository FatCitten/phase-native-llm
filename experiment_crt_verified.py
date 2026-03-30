"""
Experiment 2: CRT Composition (VERIFIED CRT FUNCTION)
=====================================================
Using verified CRT: crt(a1, m1, a2, m2) = (a1*m2*inv(m2,m1) + a2*m1*inv(m1,m2)) % (m1*m2)
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


def crt(a1, m1, a2, m2):
    """Solve x ≡ a1 (mod m1), x ≡ a2 (mod m2)"""
    import math
    if math.gcd(m1, m2) != 1:
        return None  # No unique solution
    inv_m1 = pow(m1, -1, m2)
    inv_m2 = pow(m2, -1, m1)
    M = m1 * m2
    return (a1 * m2 * inv_m2 + a2 * m1 * inv_m1) % M


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
            
            combined = crt(a, m_a, b, m_b)
            
            if combined is not None and combined == y_true[i].item():
                closures += 1
        
        closure_rate = closures / n_samples
    
    return {
        'acc_a': acc_a,
        'acc_b': acc_b,
        'closure_rate': closure_rate,
    }


def main():
    print("="*60)
    print("CRT COMPOSITION EXPERIMENTS (VERIFIED)")
    print("="*60)
    
    compositions = [
        # VALID: gcd=1, both odd primes
        {'name': 'Z_3 x Z_5 -> Z_15', 'k1': 3, 'k2': 5, 'valid': True},
        {'name': 'Z_3 x Z_7 -> Z_21', 'k1': 3, 'k2': 7, 'valid': True},
        {'name': 'Z_5 x Z_7 -> Z_35', 'k1': 5, 'k2': 7, 'valid': True},
        {'name': 'Z_3 x Z_11 -> Z_33', 'k1': 3, 'k2': 11, 'valid': True},
        # INVALID: gcd > 1
        {'name': 'Z_2 x Z_2 -> Z_4', 'k1': 2, 'k2': 2, 'valid': False},
        {'name': 'Z_3 x Z_3 -> Z_9', 'k1': 3, 'k2': 3, 'valid': False},
        # BOUNDARY: involves 2
        {'name': 'Z_2 x Z_3 -> Z_6', 'k1': 2, 'k2': 3, 'valid': None},
        {'name': 'Z_2 x Z_5 -> Z_10', 'k1': 2, 'k2': 5, 'valid': None},
        {'name': 'Z_2 x Z_7 -> Z_14', 'k1': 2, 'k2': 7, 'valid': None},
    ]
    
    n_seeds = 10
    results = {}
    
    for comp in compositions:
        print(f"\n=== {comp['name']} ===")
        
        closure_rates = []
        
        for seed in range(n_seeds):
            model_a = train_zk(comp['k1'], n_samples=1000, seed=seed)
            model_b = train_zk(comp['k2'], n_samples=1000, seed=seed+100)
            
            result = test_crt_composition(
                model_a, model_b, 
                comp['k1'], comp['k2'], comp['k1']*comp['k2'],
                n_samples=500
            )
            
            closure_rates.append(result['closure_rate'])
        
        mean_closure = np.mean(closure_rates)
        std_closure = np.std(closure_rates)
        
        results[comp['name']] = {
            'k1': comp['k1'],
            'k2': comp['k2'],
            'valid': comp['valid'],
            'closure_rates': closure_rates,
            'mean_closure': mean_closure,
            'std_closure': std_closure
        }
        
        status = "SUCCEED" if mean_closure > 0.85 else "FAIL"
        print(f"  Closure: {mean_closure:.1%} +/- {std_closure:.1%} [{status}]")
    
    print("\n" + "="*60)
    print("TABLE 1: CRT COMPOSITION")
    print("="*60)
    print("\n| Composition        | Closure     | Expected |")
    print("|-------------------|-------------|----------|")
    
    for name, r in results.items():
        exp = r['valid']
        if exp is True:
            expected = "SUCCEED"
        elif exp is False:
            expected = "FAIL"
        else:
            expected = "TEST"
        print(f"| {name:17s} | {r['mean_closure']:10.1%} | {expected:8s} |")
    
    with open('experiment_crt_verified.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\nSaved to experiment_crt_verified.json")


if __name__ == "__main__":
    main()
