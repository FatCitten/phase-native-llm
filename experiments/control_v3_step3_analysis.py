"""
STEP 3 Analysis: Distance-from-diagonal + Pattern Classification Fix
=====================================================================
Post-hoc analysis of saved heatmap results with:
1. Fixed pattern classifier (DIAGONAL_FAILURE, GEOMETRIC, MEMORIZATION, NEAR_DIAGONAL_DECAY)
2. Distance-from-diagonal table (d = |a-b|)
3. Theoretical vs observed accuracy comparison
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

def compute_heatmap(model, table, k):
    model.eval()
    with torch.no_grad():
        a_all = torch.arange(k).repeat_interleave(k)
        b_all = torch.arange(k).repeat(k)
        
        logits = model(a_all, b_all)
        preds = logits.argmax(dim=-1)
        
        true = torch.tensor([table[(a, b)] for a, b in zip(a_all.tolist(), b_all.tolist())])
        
        correct = (preds == true).view(k, k).float()
        return correct.cpu().numpy()

def compute_distance_table(heatmap, k):
    """Compute accuracy as function of distance from diagonal: d = |a-b|"""
    distance_acc = {d: [] for d in range(k)}
    
    for a in range(k):
        for b in range(k):
            d = abs(a - b)
            distance_acc[d].append(heatmap[a, b])
    
    result = {}
    for d in range(k):
        if distance_acc[d]:
            result[d] = {
                'mean': float(np.mean(distance_acc[d])),
                'count': len(distance_acc[d])
            }
    return result

def classify_pattern(diag_acc, upper_acc, lower_acc):
    """Classify the pattern based on region accuracies."""
    if upper_acc > 0.95 and lower_acc > 0.95 and diag_acc < 0.5:
        return "DIAGONAL_FAILURE"
    elif diag_acc > 0.95 and upper_acc > 0.95 and lower_acc > 0.95:
        return "GEOMETRIC"
    elif abs(upper_acc - lower_acc) < 0.1 and abs(upper_acc - diag_acc) < 0.1:
        return "MEMORIZATION"
    else:
        return "NEAR_DIAGONAL_DECAY"

def train_and_get_heatmap(table, k, train_indices, test_indices, seed, hidden_mult=1, n_epochs=150, lr=0.1):
    set_seed(seed)
    
    model = ZkBundleSimpleScaled(k, hidden_mult=hidden_mult)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    (a_tr, b_tr, y_tr), (a_te, b_te, y_te) = make_dataset_fixed(table, k, train_indices, test_indices)
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(a_tr, b_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        te_logits = model(a_te, b_te)
        te_acc = (te_logits.argmax(dim=-1) == y_te).float().mean().item()
        
        heatmap = compute_heatmap(model, table, k)
    
    return {
        'test_acc': te_acc,
        'heatmap': heatmap,
    }

def run_analysis():
    K_VALUES = [5, 7, 11]
    N_SEEDS = 10
    HIDDEN_MULT = 1
    
    RESULTS = {}
    
    print("=" * 80)
    print("STEP 3 Analysis: Distance-from-Diagonal + Fixed Pattern Classification")
    print("=" * 80)
    
    for k in K_VALUES:
        print(f"\n{'='*80}")
        print(f"k = {k}")
        print(f"{'='*80}")
        
        table = make_max_table(k)
        train_indices, test_indices = generate_split(table, k, seed=0)
        
        all_diag = []
        all_upper = []
        all_lower = []
        all_distance = {d: [] for d in range(k)}
        all_test_acc = []
        
        for seed in range(N_SEEDS):
            result = train_and_get_heatmap(table, k, train_indices, test_indices, seed, hidden_mult=HIDDEN_MULT)
            
            heatmap = result['heatmap']
            all_test_acc.append(result['test_acc'])
            
            diagonal = np.diag(heatmap)
            upper = np.triu(heatmap, k=1)
            lower = np.tril(heatmap, k=-1)
            
            diag_acc = diagonal.mean() if len(diagonal) > 0 else 0
            upper_acc = upper[upper > 0].mean() if (upper > 0).any() else 0
            lower_acc = lower[lower > 0].mean() if (lower > 0).any() else 0
            
            all_diag.append(diag_acc)
            all_upper.append(upper_acc)
            all_lower.append(lower_acc)
            
            distance_table = compute_distance_table(heatmap, k)
            for d in range(k):
                if d in distance_table:
                    all_distance[d].append(distance_table[d]['mean'])
        
        avg_diag = np.mean(all_diag)
        avg_upper = np.mean(all_upper)
        avg_lower = np.mean(all_lower)
        avg_test_acc = np.mean(all_test_acc)
        
        pattern = classify_pattern(avg_diag, avg_upper, avg_lower)
        
        print(f"\n--- Region Accuracy ---")
        print(f"Diagonal (a==b):      {avg_diag:.3f}")
        print(f"Upper triangle (a>b): {avg_upper:.3f}")
        print(f"Lower triangle (a<b): {avg_lower:.3f}")
        print(f"Overall test accuracy: {avg_test_acc:.3f}")
        print(f"\nPattern: {pattern}")
        
        print(f"\n--- Distance-from-Diagonal Table ---")
        print(f"d = |a-b|  (count):  accuracy")
        print("-" * 35)
        
        for d in range(k):
            if all_distance[d]:
                mean_acc = np.mean(all_distance[d])
                count = len(all_distance[d])
                diag_marker = "(diagonal)" if d == 0 else ""
                print(f"d={d:2d}  (n={count:2d}):  {mean_acc:.3f}  {diag_marker}")
        
        theoretical_if_only_diagonal_fails = (k*k - k) / (k*k)
        print(f"\n--- Theoretical vs Observed ---")
        print(f"If ONLY diagonal fails (0% diagonal, 100% elsewhere):")
        print(f"  Expected accuracy = {theoretical_if_only_diagonal_fails:.3f} ({theoretical_if_only_diagonal_fails*100:.1f}%)")
        print(f"  Observed accuracy = {avg_test_acc:.3f} ({avg_test_acc*100:.1f}%)")
        print(f"  Gap = {theoretical_if_only_diagonal_fails - avg_test_acc:.3f}")
        
        RESULTS[k] = {
            'pattern': pattern,
            'diagonal': float(avg_diag),
            'upper': float(avg_upper),
            'lower': float(avg_lower),
            'test_acc': float(avg_test_acc),
            'theoretical_acc_if_only_diagonal_fails': float(theoretical_if_only_diagonal_fails),
            'gap': float(theoretical_if_only_diagonal_fails - avg_test_acc),
            'distance_table': {d: float(np.mean(all_distance[d])) if all_distance[d] else None for d in range(k)},
        }
    
    Path("results/control_v3").mkdir(parents=True, exist_ok=True)
    with open("results/control_v3/step3_analysis_fixed.json", 'w') as f:
        json.dump(RESULTS, f, indent=2)
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    for k in K_VALUES:
        r = RESULTS[k]
        print(f"\nk={k}: {r['pattern']}")
        print(f"  Observed: {r['test_acc']*100:.1f}% | Theoretical (diagonal only): {r['theoretical_acc_if_only_diagonal_fails']*100:.1f}% | Gap: {r['gap']*100:.1f}%")
    
    print(f"\nResults saved to results/control_v3/step3_analysis_fixed.json")
    return RESULTS

if __name__ == "__main__":
    results = run_analysis()
