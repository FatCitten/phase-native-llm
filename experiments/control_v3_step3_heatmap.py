"""
STEP 3: max_ab Diagnostic - Per-cell accuracy heatmap
======================================================
For each seed that achieves acc > 0.5 on max_ab:
  1. Compute per-cell accuracy on the k×k input grid
  2. Print the accuracy heatmap (k rows × k columns)
  3. Check: is accuracy high on diagonal (a==b)?
            is accuracy high on one triangle (a>b or a<b)?
            or is it scattered (pure memorization)?
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import math
from itertools import product as iproduct
from pathlib import Path

# ─── REPRODUCIBILITY ────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ─── TABLE GENERATORS ───────────────────────────────────────────────────────

def make_max_table(k):
    """max(a,b)"""
    return {(a,b): max(a,b) for a,b in iproduct(range(k), range(k))}

# ─── ARCHITECTURE: ZkBundleSimpleScaled ─────────────────────────────────────

class ZkBundleSimpleScaled(nn.Module):
    """Same as step 2"""
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
    
    def get_kappa(self):
        with torch.no_grad():
            input_phases, _ = self.bundles[0]
            phases = input_phases.data
            mean_real = torch.cos(phases).mean()
            mean_imag = torch.sin(phases).mean()
            kappa = (mean_real**2 + mean_imag**2).sqrt().item()
            return kappa

# ─── DATASET ────────────────────────────────────────────────────────────────

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

# ─── HEATMAP DIAGNOSTIC ────────────────────────────────────────────────────

def compute_heatmap(model, table, k):
    """Compute per-cell accuracy on k×k grid."""
    model.eval()
    with torch.no_grad():
        # Create all k×k input pairs
        a_all = torch.arange(k).repeat_interleave(k)
        b_all = torch.arange(k).repeat(k)
        
        # Get predictions
        logits = model(a_all, b_all)
        preds = logits.argmax(dim=-1)
        
        # Get true values
        true = torch.tensor([table[(a, b)] for a, b in zip(a_all.tolist(), b_all.tolist())])
        
        # Compute accuracy per cell
        correct = (preds == true).view(k, k).float()
        return correct.cpu().numpy()

def print_heatmap(heatmap, k, title=""):
    """Print heatmap with nice formatting."""
    print(f"\n{title}")
    print("-" * 50)
    print(f"       b=0   b=1   b=2", end="")
    if k > 3:
        print("  ... ", end="")
        print(f"  b={k-1}")
    else:
        print()
    
    for i in range(k):
        row_str = f"a={i}: "
        for j in range(k):
            val = heatmap[i, j]
            if val == 1.0:
                row_str += " 1.0 "
            elif val == 0.0:
                row_str += " 0.0 "
            else:
                row_str += f"{val:.2f} "
        print(row_str)
    
    # Summary statistics
    diagonal = np.diag(heatmap)
    upper = np.triu(heatmap, k=1)
    lower = np.tril(heatmap, k=-1)
    
    diag_acc = diagonal.mean() if len(diagonal) > 0 else 0
    upper_acc = upper[upper > 0].mean() if (upper > 0).any() else 0
    lower_acc = lower[lower > 0].mean() if (lower > 0).any() else 0
    
    print("-" * 50)
    print(f"Diagonal (a==b):     {diag_acc:.3f}")
    print(f"Upper triangle (a>b): {upper_acc:.3f}")
    print(f"Lower triangle (a<b): {lower_acc:.3f}")
    
    return diag_acc, upper_acc, lower_acc

# ─── TRAINING ───────────────────────────────────────────────────────────────

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
        kappa = model.get_kappa()
        
        # Compute heatmap on FULL grid (not just test set)
        heatmap = compute_heatmap(model, table, k)
    
    return {
        'test_acc': te_acc,
        'kappa': kappa,
        'heatmap': heatmap,
    }

# ─── MAIN ──────────────────────────────────────────────────────────────────

def run_diagnostic():
    K_VALUES = [5, 7, 11]
    N_SEEDS = 10
    HIDDEN_MULT = 1  # Use original for diagnostic
    
    RESULTS = {}
    
    print("=" * 70)
    print("STEP 3: max_ab Diagnostic - Heatmap Analysis")
    print("=" * 70)
    
    for k in K_VALUES:
        print(f"\n{'='*70}")
        print(f"k = {k}")
        print(f"{'='*70}")
        
        table = make_max_table(k)
        train_indices, test_indices = generate_split(table, k, seed=0)
        
        all_diag = []
        all_upper = []
        all_lower = []
        
        for seed in range(N_SEEDS):
            result = train_and_get_heatmap(table, k, train_indices, test_indices, seed, hidden_mult=HIDDEN_MULT)
            
            print(f"\n--- Seed {seed} ---")
            print(f"Test Acc: {result['test_acc']:.3f}, Kappa: {result['kappa']:.3f}")
            
            heatmap = result['heatmap']
            diag_acc, upper_acc, lower_acc = print_heatmap(heatmap, k)
            
            all_diag.append(diag_acc)
            all_upper.append(upper_acc)
            all_lower.append(lower_acc)
        
        print(f"\n{'='*70}")
        print(f"SUMMARY for k={k}:")
        print(f"  Diagonal (a==b):     {np.mean(all_diag):.3f} +/- {np.std(all_diag):.3f}")
        print(f"  Upper (a>b):         {np.mean(all_upper):.3f} +/- {np.std(all_upper):.3f}")
        print(f"  Lower (a<b):         {np.mean(all_lower):.3f} +/- {np.std(all_lower):.3f}")
        
        # Determine pattern
        avg_diag = np.mean(all_diag)
        avg_upper = np.mean(all_upper)
        avg_lower = np.mean(all_lower)
        
        if avg_diag > avg_upper + 0.2 and avg_diag > avg_lower + 0.2:
            pattern = "DIAGONAL: Model learns a==b cases best"
        elif avg_upper > avg_lower + 0.2:
            pattern = "UPPER TRIANGLE: Model learns a>b cases better"
        elif avg_lower > avg_upper + 0.2:
            pattern = "LOWER TRIANGLE: Model learns a<b cases better"
        else:
            pattern = "MIXED/SCATTERED: No clear pattern"
        
        print(f"  Pattern: {pattern}")
        
        RESULTS[k] = {
            'diagonal': {'mean': float(np.mean(all_diag)), 'std': float(np.std(all_diag))},
            'upper': {'mean': float(np.mean(all_upper)), 'std': float(np.std(all_upper))},
            'lower': {'mean': float(np.mean(all_lower)), 'std': float(np.std(all_lower))},
            'pattern': pattern,
        }
    
    # Save results
    Path("results/control_v3").mkdir(parents=True, exist_ok=True)
    with open("results/control_v3/step3_heatmap_diagnostic.json", 'w') as f:
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k,v in obj.items()}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj
        json.dump(clean(RESULTS), f, indent=2)
    
    print(f"\nResults saved to results/control_v3/step3_heatmap_diagnostic.json")
    return RESULTS

if __name__ == "__main__":
    results = run_diagnostic()
