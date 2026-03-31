"""
Phase Spacing Regularization Experiment
=========================================
Test if preventing phase collapse recovers ceiling accuracy.

Regularization: L_reg = -lambda * min_gap_ratio
This encourages minimum gap to stay away from zero.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import math
from itertools import product as iproduct
from pathlib import Path

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_max_table(k):
    return {(a,b): max(a,b) for a,b in iproduct(range(k), range(k))}

class ZkBundleSimpleScaled(nn.Module):
    def __init__(self, k, hidden_mult=1):
        super().__init__()
        self.k = k
        self.hidden_mult = hidden_mult
        
        self.bundles = []
        for i in range(hidden_mult):
            input_phases = nn.Parameter(
                torch.tensor([j * 2 * math.pi / k for j in range(k)])
            )
            output_phases = nn.Parameter(
                torch.tensor([j * 2 * math.pi / k for j in range(k)])
            )
            self.bundles.append((input_phases, output_phases))
            self.register_parameter(f'bundle_{i}_input', input_phases)
            self.register_parameter(f'bundle_{i}_output', output_phases)
    
    def forward(self, x1, x2):
        all_logits = []
        for input_phases, output_phases in self.bundles:
            p1 = input_phases[x1]
            p2 = input_phases[x2]
            phi = (p1 + p2) % (2 * math.pi)
            dists = torch.abs(phi.unsqueeze(-1) - output_phases.unsqueeze(0))
            dists = dists % (2 * math.pi)
            dists = torch.min(dists, 2 * math.pi - dists)
            logits = -dists
            all_logits.append(logits)
        result = torch.stack(all_logits, dim=0).mean(dim=0)
        return result
    
    def get_output_phases(self):
        """Return output phases from first bundle."""
        return self.bundles[0][1]
    
    def compute_min_gap_ratio(self):
        """Compute min_gap / theoretical_gap for output phases."""
        phases = self.get_output_phases()
        k = self.k
        theoretical_gap = 2 * math.pi / k
        
        # Sort phases and compute gaps
        phases_sorted = torch.sort(phases % (2 * math.pi))[0]
        gaps = []
        for i in range(k):
            next_i = (i + 1) % k
            gap = (phases_sorted[next_i] - phases_sorted[i]) % (2 * math.pi)
            if gap < 1e-10:
                gap = (2 * math.pi - phases_sorted[i]) + phases_sorted[next_i]
            gaps.append(gap)
        
        min_gap = min(gaps)
        return min_gap / theoretical_gap

def make_dataset_fixed(table, k, train_indices, test_indices):
    pairs = list(table.keys())
    def to_tensors(indices):
        ps = [pairs[i] for i in indices]
        a = torch.tensor([p[0] for p in ps], dtype=torch.long)
        b = torch.tensor([p[1] for p in ps], dtype=torch.long)
        y = torch.tensor([table[p] for p in ps], dtype=torch.long)
        return a, b, y
    return to_tensors(train_indices), to_tensors(test_indices)

def generate_split(table, k, seed=0):
    rng = random.Random(seed)
    pairs = list(table.keys())
    pairs_copy = pairs.copy()
    rng.shuffle(pairs_copy)
    split = int(0.8 * len(pairs_copy))
    return list(range(split)), list(range(split, len(pairs_copy)))

def compute_ceiling(model, table, k, n_samples=500):
    """Estimate ceiling accuracy by sampling many pairs."""
    model.eval()
    all_pairs = list(table.keys())
    
    if len(all_pairs) <= n_samples:
        pairs_to_test = all_pairs
    else:
        pairs_to_test = random.sample(all_pairs, n_samples)
    
    correct = 0
    total = 0
    for a, b in pairs_to_test:
        with torch.no_grad():
            a_t = torch.tensor([a])
            b_t = torch.tensor([b])
            pred = model(a_t, b_t).argmax().item()
            true = table[(a, b)]
            if pred == true:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0

def train_with_reg(table, k, seed, lambda_reg=0.0, hidden_mult=1, n_epochs=150, lr=0.1):
    """Train with phase spacing regularization."""
    set_seed(seed)
    
    train_indices, test_indices = generate_split(table, k, seed)
    
    model = ZkBundleSimpleScaled(k, hidden_mult=hidden_mult)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    (a_tr, b_tr, y_tr), (a_te, b_te, y_te) = make_dataset_fixed(table, k, train_indices, test_indices)
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Standard loss
        logits = model(a_tr, b_tr)
        loss_ce = criterion(logits, y_tr)
        
        # Regularization: penalize small min_gap_ratio
        if lambda_reg > 0:
            min_gap_ratio = model.compute_min_gap_ratio()
            # Regularization: want min_gap_ratio to be large
            # L_reg = -lambda * min_gap_ratio
            loss_reg = -lambda_reg * min_gap_ratio
            loss = loss_ce + loss_reg
        else:
            loss = loss_ce
        
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    
    # Training accuracy
    with torch.no_grad():
        train_logits = model(a_tr, b_tr)
        train_acc = (train_logits.argmax(dim=-1) == y_tr).float().mean().item()
        
        test_logits = model(a_te, b_te)
        test_acc = (test_logits.argmax(dim=-1) == y_te).float().mean().item()
    
    # Ceiling accuracy (full evaluation)
    ceiling_acc = compute_ceiling(model, table, k)
    
    # Min gap ratio
    min_gap_ratio = model.compute_min_gap_ratio().item()
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'ceiling_acc': ceiling_acc,
        'min_gap_ratio': min_gap_ratio,
        'lambda_reg': lambda_reg
    }

# ============================================================================
# MAIN
# ============================================================================
print("=" * 70)
print("PHASE SPACING REGULARIZATION EXPERIMENT")
print("=" * 70)

# Baseline results from previous experiments
baseline_results = {
    19: {'ceiling': 0.842, 'min_gap': 0.074},
    23: {'ceiling': 0.826, 'min_gap': 0.044},
    29: {'ceiling': 0.690, 'min_gap': 0.005}
}

k_values = [19, 23, 29]
lambda_values = [0.0, 0.01, 0.1]
seeds = [0, 1, 2]  # Multiple seeds for robustness

results = []

for k in k_values:
    print(f"\n=== k = {k} ===")
    
    table = make_max_table(k)
    
    for lambda_reg in lambda_values:
        print(f"  lambda = {lambda_reg}")
        
        for seed in seeds:
            result = train_with_reg(
                table, k, seed,
                lambda_reg=lambda_reg,
                hidden_mult=1,
                n_epochs=200,
                lr=0.1
            )
            
            results.append({
                'k': k,
                'lambda': lambda_reg,
                'seed': seed,
                'ceiling_acc': result['ceiling_acc'],
                'min_gap_ratio': result['min_gap_ratio'],
                'test_acc': result['test_acc']
            })
            
            print(f"    seed={seed}: ceiling={result['ceiling_acc']:.3f}, min_gap={result['min_gap_ratio']:.3f}")

# Aggregate results
print("\n" + "=" * 70)
print("AGGREGATED RESULTS")
print("=" * 70)

print(f"\n{'k':>4} | {'lambda':>8} | {'mean_ceiling':>12} | {'mean_min_gap':>12} | {'n_seeds':>8}")
print("-" * 60)

for k in k_values:
    for lambda_reg in lambda_values:
        subset = [r for r in results if r['k'] == k and r['lambda'] == lambda_reg]
        mean_ceiling = np.mean([r['ceiling_acc'] for r in subset])
        mean_min_gap = np.mean([r['min_gap_ratio'] for r in subset])
        
        baseline_ceiling = baseline_results[k]['ceiling']
        delta = mean_ceiling - baseline_ceiling
        
        print(f"{k:>4} | {lambda_reg:>8.2f} | {mean_ceiling:>12.3f} | {mean_min_gap:>12.3f} | {len(subset):>8}  (delta: {delta:+.3f})")

# Save results
Path("results/ceiling_decay").mkdir(parents=True, exist_ok=True)
with open("results/ceiling_decay/regularization_results.json", 'w') as f:
    json.dump({
        "experiment": "phase_spacing_regularization",
        "results": results,
        "baseline": baseline_results
    }, f, indent=2)

print("\nSaved: results/ceiling_decay/regularization_results.json")
