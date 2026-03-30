"""
NEW-E: Z_6 Composite Test
=========================
Test Z_6 (k = 6 = 2 x 3, composite but NOT prime power).

Two competing predictions:
- If rule is "prime order only"      -> Z_6 FAILS
- If rule is "not prime-power"        -> Z_6 SUCCEEDS

Result discriminates between these two hypotheses.
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


def generate_zk_data(k, n_samples):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    return x1, x2, y


def train_zk(k, n_samples, epochs=150, seed=42):
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
    
    return model, x1, x2, y


def evaluate_accuracy(model, x1, x2, y):
    with torch.no_grad():
        outputs = model(x1, x2)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    return accuracy


def main():
    print("="*60)
    print("NEW-E: Z_6 COMPOSITE TEST")
    print("="*60)
    print("\nk = 6 = 2 x 3 (composite, NOT prime power)")
    print("\nTwo predictions:")
    print("  - 'Prime order only' rule: Z_6 FAILS")
    print("  - 'Not prime-power' rule:   Z_6 SUCCEEDS")
    print("="*60)
    
    k = 6
    n_seeds = 10
    n_samples = 1000
    epochs = 150
    
    results = []
    
    for seed in range(n_seeds):
        model, x1, x2, y = train_zk(k, n_samples, epochs=epochs, seed=seed)
        accuracy = evaluate_accuracy(model, x1, x2, y)
        
        result = {
            'seed': seed,
            'accuracy': accuracy,
            'z6_passes': accuracy >= 0.8
        }
        results.append(result)
        
        print(f"  Seed {seed}: accuracy = {accuracy:.4f} ({'PASS' if result['z6_passes'] else 'FAIL'})")
    
    n_passed = sum(1 for r in results if r['z6_passes'])
    mean_accuracy = np.mean([r['accuracy'] for r in results])
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"k = {k} (composite: 2 x 3)")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Pass rate: {n_passed}/{n_seeds}")
    
    if n_passed >= n_seeds // 2:
        print("\n*** RESULT: Z_6 SUCCEEDS ***")
        print("Supports 'not prime-power' rule")
        print("Z_6 works because 6 is NOT a prime power (2^n)")
    else:
        print("\n*** RESULT: Z_6 FAILS ***")
        print("Supports 'prime order only' rule")
        print("Wait - this is unexpected for non-prime-power!")
    
    output = {
        'k': k,
        'factorization': '2 x 3',
        'is_prime_power': False,
        'n_seeds': n_seeds,
        'n_samples': n_samples,
        'epochs': epochs,
        'mean_accuracy': mean_accuracy,
        'n_passed': n_passed,
        'pass_rate': n_passed / n_seeds,
        'results': results
    }
    
    with open('experiment_new_e_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to experiment_new_e_results.json")
    
    return output


if __name__ == "__main__":
    main()
