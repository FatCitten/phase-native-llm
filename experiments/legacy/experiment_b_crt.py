"""
Experiment B: CRT Composition Test
================================
Train Z_3 and Z_5 separately, then compose to solve Z_15 COLD.
Test if geometric composition works without retraining.
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
    
    def get_phases(self):
        return self.input_phases.detach().cpu().numpy()
    
    def get_output_phases(self):
        return self.output_phases.detach().cpu().numpy()


def extended_gcd(a, b):
    if b == 0:
        return (1, 0, a)
    else:
        x1, y1, g = extended_gcd(b, a % b)
        return (y1, x1 - (a // b) * y1, g)


def crt_reconstruct(a3, a5):
    m1, m2 = 3, 5
    inv1 = extended_gcd(m1, m2)[0] % m2
    inv2 = extended_gcd(m2, m1)[0] % m1
    x = (a3 * m2 * inv1 + a5 * m1 * inv2) % (m1 * m2)
    return x


def generate_z3_data(n_samples):
    x1 = torch.randint(0, 3, (n_samples,))
    x2 = torch.randint(0, 3, (n_samples,))
    y = (x1 + x2) % 3
    return x1, x2, y


def generate_z5_data(n_samples):
    x1 = torch.randint(0, 5, (n_samples,))
    x2 = torch.randint(0, 5, (n_samples,))
    y = (x1 + x2) % 5
    return x1, x2, y


def generate_z15_data(n_samples):
    x1 = torch.randint(0, 15, (n_samples,))
    x2 = torch.randint(0, 15, (n_samples,))
    y = (x1 + x2) % 15
    return x1, x2, y


def train_zk(k, n_samples=1000, epochs=150, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if k == 3:
        x1, x2, y = generate_z3_data(n_samples)
    elif k == 5:
        x1, x2, y = generate_z5_data(n_samples)
    else:
        raise ValueError(f"Unsupported k: {k}")
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model


def test_zk_accuracy(model, k, n_samples=500):
    if k == 3:
        x1, x2, y = generate_z3_data(n_samples)
    elif k == 5:
        x1, x2, y = generate_z5_data(n_samples)
    else:
        raise ValueError(f"Unsupported k: {k}")
    
    with torch.no_grad():
        outputs = model(x1, x2)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    return accuracy


def test_crt_composition(model3, model5, n_samples=500):
    x1, x2, y_true = generate_z15_data(n_samples)
    
    x1_mod3 = x1 % 3
    x2_mod3 = x2 % 3
    x1_mod5 = x1 % 5
    x2_mod5 = x2 % 5
    
    y_mod3_true = (x1_mod3 + x2_mod3) % 3
    y_mod5_true = (x1_mod5 + x2_mod5) % 5
    
    with torch.no_grad():
        outputs_mod3 = model3(x1_mod3, x2_mod3)
        outputs_mod5 = model5(x1_mod5, x2_mod5)
        
        pred_mod3 = outputs_mod3.argmax(1)
        pred_mod5 = outputs_mod5.argmax(1)
        
        correct_mod3 = (pred_mod3 == y_mod3_true).float().mean().item()
        correct_mod5 = (pred_mod5 == y_mod5_true).float().mean().item()
        
        closures = []
        for i in range(n_samples):
            a3 = pred_mod3[i].item()
            a5 = pred_mod5[i].item()
            y_pred_crt = crt_reconstruct(a3, a5)
            closures.append(y_pred_crt == y_true[i].item())
        
        closure_rate = np.mean(closures)
    
    return {
        'mod3_accuracy': correct_mod3,
        'mod5_accuracy': correct_mod5,
        'crt_closure_rate': closure_rate,
        'total_correct': sum(closures),
        'total_samples': n_samples
    }


def measure_holonomy_closure(model3, model5, n_samples=500):
    x1, x2, y_true = generate_z15_data(n_samples)
    
    x1_mod3 = x1 % 3
    x2_mod3 = x2 % 3
    x1_mod5 = x1 % 5
    x2_mod5 = x2 % 5
    
    y_mod3_true = (x1_mod3 + x2_mod3) % 3
    y_mod5_true = (x1_mod5 + x2_mod5) % 5
    
    with torch.no_grad():
        outputs_mod3 = model3(x1_mod3, x2_mod3)
        outputs_mod5 = model5(x1_mod5, x2_mod5)
        
        pred_mod3 = outputs_mod3.argmax(1)
        pred_mod5 = outputs_mod5.argmax(1)
        
        correct_mod3 = (pred_mod3 == y_mod3_true).sum().item()
        correct_mod5 = (pred_mod5 == y_mod5_true).sum().item()
        
        closures = 0
        for i in range(n_samples):
            a3 = pred_mod3[i].item()
            a5 = pred_mod5[i].item()
            y_pred_crt = crt_reconstruct(a3, a5)
            if y_pred_crt == y_true[i].item():
                closures += 1
        
        correct_both = (correct_mod3 == n_samples and correct_mod5 == n_samples)
        
        holonomy_closes = correct_both and closures > 0
        
        return {
            'holonomy_closes': holonomy_closes,
            'mod3_correct': correct_mod3,
            'mod5_correct': correct_mod5,
            'crt_closure': closures,
            'total': n_samples
        }


def main():
    print("="*60)
    print("EXPERIMENT B: CRT COMPOSITION TEST")
    print("Train Z_3 and Z_5 separately -> compose for Z_15")
    print("="*60)
    
    n_seeds = 10
    n_test = 500
    
    results = []
    
    for seed in range(n_seeds):
        print(f"\n=== Seed {seed} ===")
        
        model3 = train_zk(3, n_samples=1000, seed=seed)
        model5 = train_zk(5, n_samples=1000, seed=seed+100)
        
        acc3 = test_zk_accuracy(model3, 3, n_test)
        acc5 = test_zk_accuracy(model5, 5, n_test)
        
        print(f"  Z_3 accuracy: {acc3:.2%}")
        print(f"  Z_5 accuracy: {acc5:.2%}")
        
        crt_result = test_crt_composition(model3, model5, n_test)
        
        holonomy = measure_holonomy_closure(model3, model5, n_test)
        
        result = {
            'seed': seed,
            'z3_accuracy': acc3,
            'z5_accuracy': acc5,
            'crt_closure_rate': crt_result['crt_closure_rate'],
            'holonomy_closes': holonomy['holonomy_closes'],
            'mod3_correct': holonomy['mod3_correct'],
            'mod5_correct': holonomy['mod5_correct'],
            'crt_correct': holonomy['crt_closure']
        }
        results.append(result)
        
        print(f"  CRT closure rate: {crt_result['crt_closure_rate']:.2%}")
        print(f"  Holonomy closes: {holonomy['holonomy_closes']}")
    
    closure_rates = [r['crt_closure_rate'] for r in results]
    holonomy_closes = sum([r['holonomy_closes'] for r in results])
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nMean CRT closure rate: {np.mean(closure_rates):.2%}")
    print(f"Holonomy closes: {holonomy_closes}/{n_seeds} seeds")
    
    if holonomy_closes / n_seeds > 0.9:
        print("\n*** CRT EMERGENCE CONFIRMED ***")
        print("Geometric composition works without retraining!")
    else:
        print("\n*** CRT EMERGENCE NOT CONFIRMED ***")
        print("Network did not discover CRT structure from fiber + bundle alone.")
    
    with open('experiment_b_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiment_b_results.json")
    
    return results


if __name__ == "__main__":
    main()
