"""
Control v3 Step 8: Retrain k=11 Fresh
======================================
Purpose: Verify undertraining hypothesis
- Original k=11 gave theta=0.141
- k=13,17 gave theta=0.318, 0.452
- If fresh k=11 gives theta≈0.30, original was undertrained
- If theta≈0.14, the jump is real
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import math
import os
from scipy.optimize import curve_fit
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

def compute_wrapped_distance_table(heatmap, k):
    """Compute accuracy by wrapped distance: min(d, k-d)"""
    wrapped_acc = {d: [] for d in range(k // 2 + 1)}
    
    for a in range(k):
        for b in range(k):
            d_raw = abs(a - b)
            wrapped_d = min(d_raw, k - d_raw)
            wrapped_acc[wrapped_d].append(heatmap[a, b])
    
    result = {}
    for d in range(k // 2 + 1):
        if wrapped_acc[d]:
            result[d] = {
                'mean': float(np.mean(wrapped_acc[d])),
                'count': len(wrapped_acc[d])
            }
    return result

def compute_raw_distance_table(heatmap, k):
    """Compute accuracy by raw distance: d = |a-b| (no wrapping)"""
    raw_acc = {d: [] for d in range(k)}
    
    for a in range(k):
        for b in range(k):
            d_raw = abs(a - b)
            raw_acc[d_raw].append(heatmap[a, b])
    
    result = {}
    for d in range(k):
        if raw_acc[d]:
            result[d] = {
                'mean': float(np.mean(raw_acc[d])),
                'count': len(raw_acc[d])
            }
    return result

def sigmoid(x, theta, beta):
    return 1.0 / (1.0 + np.exp(-(x - theta) / beta))

def fit_sigmoid(x_data, y_data):
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    try:
        popt, _ = curve_fit(sigmoid, x_arr, y_arr, p0=[0.3, 0.1], bounds=([-0.5, 0.01], [1.0, 0.5]))
        theta, beta = popt
        y_pred = sigmoid(x_arr, theta, beta)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return float(theta), float(beta), float(r_squared)
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None, None

def train_and_evaluate(table, k, seed, hidden_mult=1, n_epochs=150, lr=0.1):
    set_seed(seed)
    
    train_indices, test_indices = generate_split(table, k, seed)
    
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
        tr_logits = model(a_tr, b_tr)
        tr_preds = tr_logits.argmax(dim=-1)
        tr_acc = (tr_preds == y_tr).float().mean().item()
        
        te_logits = model(a_te, b_te)
        te_preds = te_logits.argmax(dim=-1)
        te_acc = (te_preds == y_te).float().mean().item()
    
    heatmap = compute_heatmap(model, table, k)
    wrapped_table = compute_wrapped_distance_table(heatmap, k)
    raw_table = compute_raw_distance_table(heatmap, k)
    
    return {
        'train_acc': tr_acc,
        'test_acc': te_acc,
        'heatmap': heatmap.tolist(),
        'wrapped_distance_table': wrapped_table,
        'raw_distance_table': raw_table,
        'model': model
    }

def main():
    k = 11
    n_seeds = 10
    n_epochs = 150
    lr = 0.1
    hidden_mult = 1
    
    print(f"=== Retraining k={k} fresh ===")
    print(f"Seeds: {n_seeds}, Epochs: {n_epochs}, LR: {lr}")
    
    table = make_max_table(k)
    
    model_dir = Path('results/control_v3/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for seed in range(n_seeds):
        print(f"\n  Seed {seed}/{n_seeds-1}...")
        
        result = train_and_evaluate(table, k, seed, hidden_mult=hidden_mult, n_epochs=n_epochs, lr=lr)
        
        model_path = model_dir / f'k{k}_seed{seed}.pt'
        torch.save({
            'seed': seed,
            'k': k,
            'state_dict': result['model'].state_dict(),
            'train_acc': result['train_acc'],
            'test_acc': result['test_acc']
        }, model_path)
        
        results.append({
            'seed': seed,
            'train_acc': result['train_acc'],
            'test_acc': result['test_acc'],
            'wrapped_distance_table': result['wrapped_distance_table'],
            'raw_distance_table': result['raw_distance_table']
        })
        
        print(f"    Train: {result['train_acc']:.4f}, Test: {result['test_acc']:.4f}")
    
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    print(f"\n=== Aggregate Results ===")
    print(f"Train: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"Test:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    
    wrapped_distances = []
    wrapped_accuracies = []
    for r in results:
        for d, vals in r['wrapped_distance_table'].items():
            wrapped_distances.append(d)
            wrapped_accuracies.append(vals['mean'])
    
    unique_ds = sorted(set(wrapped_distances))
    mean_acc_per_d = []
    for d in unique_ds:
        accs = [v['mean'] for r in results for dd, v in r['wrapped_distance_table'].items() if dd == d]
        mean_acc_per_d.append(np.mean(accs))
    
    x_data = [2 * d / k for d in unique_ds]
    theta, beta, r_squared = fit_sigmoid(x_data, mean_acc_per_d)
    
    print(f"\n=== Sigmoid Fit (Wrapped Normalization) ===")
    print(f"theta: {theta:.4f}")
    print(f"beta:  {beta:.4f}")
    print(f"R²:    {r_squared:.4f}")
    
    print(f"\n=== Comparison ===")
    print(f"Original k=11: theta=0.141")
    print(f"Fresh k=11:    theta={theta:.4f}")
    print(f"k=13:          theta=0.318")
    print(f"k=17:          theta=0.452")
    
    if theta > 0.25:
        print("\n==> Fresh k=11 matches k=13 trend ==> ORIGINAL WAS UNDERTRAINED")
    else:
        print("\n==> Fresh k=11 matches original ==> JUMP AT k=13 IS REAL")
    
    output = {
        'k': k,
        'n_seeds': n_seeds,
        'n_epochs': n_epochs,
        'lr': lr,
        'train_acc_mean': float(np.mean(train_accs)),
        'train_acc_std': float(np.std(train_accs)),
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'theta': theta,
        'beta': beta,
        'r_squared': r_squared,
        'wrapped_distance_points': {str(d): {'mean': float(np.mean([v['mean'] for r in results for dd, v in r['wrapped_distance_table'].items() if dd == d])), 'count': sum(1 for r in results for dd in r['wrapped_distance_table'].keys() if dd == d)} for d in unique_ds},
        'per_seed_results': results
    }
    
    output_path = f'results/control_v3/step8_k11_retrain.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    print(f"Models saved to {model_dir}")

if __name__ == '__main__':
    main()
