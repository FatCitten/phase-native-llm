"""
STEP 1: Fix mul_mod_k variance
===============================
Fix the train/test split to be identical across seeds for deterministic LUTs.

The issue: make_dataset() shuffles data after set_seed() is called,
so different seeds get different train/test splits even for deterministic functions.

Fix: Pre-generate train/test split ONCE per (k, condition), then reuse for all seeds.
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

def make_mul_table(k):
    """(a * b) mod k — multiplicative structure"""
    return {(a,b): (a*b) % k for a,b in iproduct(range(k), range(k))}

# ─── ARCHITECTURE ───────────────────────────────────────────────────────────

class ZkBundleSimple(nn.Module):
    """Original phase-native architecture."""
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
    
    def get_kappa(self):
        with torch.no_grad():
            phases = self.input_phases.data
            mean_real = torch.cos(phases).mean()
            mean_imag = torch.sin(phases).mean()
            kappa = (mean_real**2 + mean_imag**2).sqrt().item()
            return kappa

# ─── DATASET (FIXED: pre-generate split) ───────────────────────────────────

def make_dataset_fixed(table, k, train_indices, test_indices):
    """Convert lookup table to tensors using PRE-FIXED train/test split."""
    pairs = list(table.keys())
    
    def to_tensors(indices):
        ps = [pairs[i] for i in indices]
        a = torch.tensor([p[0] for p in ps], dtype=torch.long)
        b = torch.tensor([p[1] for p in ps], dtype=torch.long)
        y = torch.tensor([table[p] for p in ps], dtype=torch.long)
        return a, b, y
    
    return to_tensors(train_indices), to_tensors(test_indices)

def generate_split(table, k, seed=0):
    """Generate fixed 80/20 train/test split using a fixed seed."""
    rng = random.Random(seed)
    pairs = list(table.keys())
    pairs_copy = pairs.copy()
    rng.shuffle(pairs_copy)
    
    split = int(0.8 * len(pairs_copy))
    train_indices = list(range(split))
    test_indices = list(range(split, len(pairs_copy)))
    return train_indices, test_indices

# ─── TRAINING ───────────────────────────────────────────────────────────────

def train_one_run_fixed(table, k, train_indices, test_indices, seed, n_epochs=150, lr=0.1):
    """Single training run with FIXED train/test split."""
    set_seed(seed)
    
    model = ZkBundleSimple(k)
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
        tr_acc = (tr_logits.argmax(dim=-1) == y_tr).float().mean().item()
        
        te_logits = model(a_te, b_te)
        te_acc = (te_logits.argmax(dim=-1) == y_te).float().mean().item()
        
        kappa = model.get_kappa()
    
    return {
        'train_acc': tr_acc,
        'test_acc': te_acc,
        'kappa': kappa,
        'chance_level': 1.0 / k,
    }

# ─── MAIN ──────────────────────────────────────────────────────────────────

def run_step1():
    K_VALUES = [5, 7, 11]
    N_SEEDS = 10
    RESULTS = {}
    
    print("=" * 70)
    print("STEP 1: mul_mod_k variance test (FIXED split)")
    print("=" * 70)
    
    for k in K_VALUES:
        chance = 1.0 / k
        table = make_mul_table(k)
        
        # FIXED: Generate split ONCE, reuse for all seeds
        train_indices, test_indices = generate_split(table, k, seed=0)
        
        print(f"\nk={k} | mul_mod_k | chance={chance:.3f}")
        print(f"  Train size: {len(train_indices)}, Test size: {len(test_indices)}")
        print(f"  seed: ", end="", flush=True)
        
        seed_results = []
        for seed in range(N_SEEDS):
            print(f"{seed}", end=" ", flush=True)
            result = train_one_run_fixed(table, k, train_indices, test_indices, seed)
            seed_results.append(result)
        
        test_accs = [r['test_acc'] for r in seed_results]
        kappas = [r['kappa'] for r in seed_results]
        
        mean_acc = float(np.mean(test_accs))
        std_acc = float(np.std(test_accs))
        mean_kappa = float(np.mean(kappas))
        
        print(f"\n  acc={mean_acc:.3f}+/-{std_acc:.3f}  kappa={mean_kappa:.3f}")
        
        RESULTS[k] = {
            'test_acc_mean': mean_acc,
            'test_acc_std': std_acc,
            'kappa_mean': mean_kappa,
            'kappa_std': float(np.std(kappas)),
            'chance_level': chance,
            'raw': seed_results,
        }
        
        # Compare to old high variance
        old_std = {5: 0.328, 7: 0.189, 11: 0.289}
        print(f"  OLD std: {old_std[k]:.3f} -> NEW std: {std_acc:.3f}")
    
    # Save results
    Path("results/control_v3").mkdir(parents=True, exist_ok=True)
    with open("results/control_v3/step1_mul_mod_k.json", 'w') as f:
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k,v in obj.items()}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj
        json.dump(clean(RESULTS), f, indent=2)
    
    print(f"\nResults saved to results/control_v3/step1_mul_mod_k.json")
    return RESULTS

if __name__ == "__main__":
    results = run_step1()
