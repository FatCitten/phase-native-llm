"""
THREE-CONDITION CONTROL EXPERIMENT V2
====================================
Updated based on findings from v1.

FIX 1: Renamed max_mod_k -> max_ab (mod is vacuous for inputs in 0..k-1)
FIX 2: Added mul_mod_k and quad_mod_k conditions
FIX 3: FLIPPED kappa interpretation:
    - LOW kappa  = uniform spacing = GEOMETRIC solution = GOOD
    - HIGH kappa = clustered phases = MEMORIZATION = BAD
FIX 4 (v3): Ceiling test using SAME architecture (ZkBundleSimple) with scaled hidden_dim
           Compare [1x, 2x, 4x, 8x] capacity for max_ab and quad_mod_k

Conditions:
1. addition       - group structure (Z_k)       - expect 100%, LOW kappa
2. random_lut    - no structure                 - expect ~chance, HIGH kappa  
3. max_ab        - lattice (max), NOT group     - expect ~60%, HIGH kappa
4. mul_mod_k     - multiplicative, not group    - TESTING
5. quad_mod_k    - deterministic, no group       - TESTING

Run: python experiments/valid/random_lut_control_v2.py
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

def make_addition_table(k):
    """Z_k addition — true group structure"""
    return {(a,b): (a+b) % k for a,b in iproduct(range(k), range(k))}

def make_random_lut(k, seed=42):
    """Completely random mapping — no structure whatsoever
    
    FIXED seed=42 for ALL k, ALL seeds.
    """
    rng = random.Random(seed)
    return {(a,b): rng.randint(0, k-1) for a,b in iproduct(range(k), range(k))}

def make_max_table(k):
    """max(a,b) — lattice structure, NOT a group
    
    NOTE: max(a,b) mod k = max(a,b) when a,b ∈ {0,...,k-1}
    because max(a,b) ≤ k-1 < k, so mod NEVER fires.
    This tests lattice/ordering structure.
    """
    return {(a,b): max(a,b) for a,b in iproduct(range(k), range(k))}

def make_mul_table(k):
    """(a * b) mod k — multiplicative structure
    
    Properties:
    - Commutative: a*b = b*a
    - NOT a group when k is composite (non-coprime elements have no inverse)
    - For prime k: (Z_k*, *) forms a group
    """
    return {(a,b): (a*b) % k for a,b in iproduct(range(k), range(k))}

def make_quad_table(k):
    """(a*a + b) mod k — deterministic, NO group structure
    
    Quadratic form + linear term.
    Fully deterministic, MLP-learnable, zero group structure.
    """
    return {(a,b): (a*a + b) % k for a,b in iproduct(range(k), range(k))}

# ─── ARCHITECTURE ───────────────────────────────────────────────────────────

class ZkBundleSimple(nn.Module):
    """
    Phase-native architecture with optional hidden layers.
    Uses phase addition as the core operation, with optional MLP capacity.
    
    hidden_dim_multiplier: 1=original (no hidden), 2/4/8 for capacity tests
    """
    def __init__(self, k, hidden_dim_multiplier=1, base_hidden=16):
        super().__init__()
        self.k = k
        self.hidden_dim = base_hidden * hidden_dim_multiplier
        self.hidden_dim_multiplier = hidden_dim_multiplier
        
        self.input_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
        self.output_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
        
        if hidden_dim_multiplier > 1:
            self.net = nn.Sequential(
                nn.Linear(2*k, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, k)
            )
        else:
            self.net = None
    
    def forward(self, x1, x2):
        if self.net is None:
            p1 = self.input_phases[x1]
            p2 = self.input_phases[x2]
            phi = (p1 + p2) % (2 * math.pi)
            dists = torch.abs(phi.unsqueeze(-1) - self.output_phases.unsqueeze(0))
            dists = dists % (2 * math.pi)
            dists = torch.min(dists, 2 * math.pi - dists)
            return -dists
        else:
            x1_oh = torch.nn.functional.one_hot(x1, num_classes=self.k).float()
            x2_oh = torch.nn.functional.one_hot(x2, num_classes=self.k).float()
            combined = torch.cat([x1_oh, x2_oh], dim=-1)
            return self.net(combined)
    
    def get_kappa(self):
        """Compute phase coherence (low = geometric, high = memorization)."""
        with torch.no_grad():
            phases = self.input_phases.data
            mean_real = torch.cos(phases).mean()
            mean_imag = torch.sin(phases).mean()
            kappa = (mean_real**2 + mean_imag**2).sqrt().item()
            return kappa

# ─── DATASET ─────────────────────────────────────────────────────────────────

def make_dataset(table, k):
    """Convert lookup table to tensors. 80/20 train/test split."""
    pairs = list(table.keys())
    random.shuffle(pairs)
    
    split = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    test_pairs = pairs[split:]
    
    def to_tensors(ps):
        a = torch.tensor([p[0] for p in ps], dtype=torch.long)
        b = torch.tensor([p[1] for p in ps], dtype=torch.long)
        y = torch.tensor([table[p] for p in ps], dtype=torch.long)
        return a, b, y
    
    return to_tensors(train_pairs), to_tensors(test_pairs)

# ─── TRAINING ───────────────────────────────────────────────────────────────

def train_one_run(table, k, seed, hidden_dim_mult=1, n_epochs=150, lr=0.1):
    """Single training run with proven hyperparameters.
    
    Args:
        table: lookup table
        k: modulus
        seed: random seed
        hidden_dim_mult: capacity multiplier (1=original, 2/4/8 for ceiling test)
        n_epochs: training epochs
        lr: learning rate
    """
    set_seed(seed)
    
    model = ZkBundleSimple(k, hidden_dim_multiplier=hidden_dim_mult)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    (a_tr, b_tr, y_tr), (a_te, b_te, y_te) = make_dataset(table, k)
    
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

# ─── MAIN EXPERIMENT LOOP ───────────────────────────────────────────────────

def run_experiment():
    K_VALUES = [5, 7, 11]
    N_SEEDS = 10
    RESULTS = {}
    
    CONDITIONS = {
        'addition': make_addition_table,
        'random_lut': lambda k: make_random_lut(k, seed=42),
        'max_ab': make_max_table,
        'mul_mod_k': make_mul_table,
        'quad_mod_k': make_quad_table,
    }
    
    CEILING_TEST_CONDS = ['max_ab', 'quad_mod_k']
    HIDDEN_SIZES = [1, 2, 4, 8]
    
    print("=" * 70)
    print("THREE-CONDITION CONTROL EXPERIMENT V2")
    print("=" * 70)
    print(f"k values:   {K_VALUES}")
    print(f"Seeds:      {N_SEEDS}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Ceiling test on: {CEILING_TEST_CONDS} with hidden_dim_mult={HIDDEN_SIZES}")
    print("=" * 70)
    
    for k in K_VALUES:
        RESULTS[k] = {}
        chance = 1.0 / k
        
        for cond_name, table_fn in CONDITIONS.items():
            sizes_to_run = HIDDEN_SIZES if cond_name in CEILING_TEST_CONDS else [1]
            
            for hidden_mult in sizes_to_run:
                size_label = f"_{hidden_mult}x" if hidden_mult > 1 else ""
                full_cond_name = cond_name + size_label
                table = table_fn(k)
                seed_results = []
                
                print(f"\nk={k:2d} | {full_cond_name:15s} | chance={chance:.3f}")
                print(f"       seed: ", end="", flush=True)
                
                for seed in range(N_SEEDS):
                    print(f"{seed}", end=" ", flush=True)
                    result = train_one_run(table, k, seed=seed, hidden_dim_mult=hidden_mult)
                    seed_results.append(result)
                
                test_accs = [r['test_acc'] for r in seed_results]
                kappas = [r['kappa'] for r in seed_results]
                
                summary = {
                    'test_acc_mean': float(np.mean(test_accs)),
                    'test_acc_std': float(np.std(test_accs)),
                    'kappa_mean': float(np.mean(kappas)),
                    'kappa_std': float(np.std(kappas)),
                    'chance_level': chance,
                    'above_chance': float(np.mean(test_accs)) > chance * 1.5,
                    'hidden_dim_mult': hidden_mult,
                    'raw': seed_results,
                }
                
                RESULTS[k][full_cond_name] = summary
                
                kappa_status = "LOW-GOOD" if summary['kappa_mean'] < 0.1 else "HIGH-BAD"
                print(f"\n       acc={summary['test_acc_mean']:.3f}+/-{summary['test_acc_std']:.3f}"
                      f"  kappa={summary['kappa_mean']:.3f}+/-{summary['kappa_std']:.3f} ({kappa_status})"
                      f"  {'PASS' if summary['above_chance'] else 'FAIL'}")
    
    return RESULTS

# ─── REPORTING ───────────────────────────────────────────────────────────────

def print_final_report(results):
    print("\n")
    print("=" * 80)
    print("FINAL RESULTS TABLE")
    print("=" * 80)
    print(f"{'k':>3} | {'Condition':15} | {'Test Acc':>10} | {'kappa':>12} | {'Status':>12}")
    print("-" * 80)
    
    for k in sorted(results.keys()):
        for cond in sorted(results[k].keys()):
            r = results[k][cond]
            if r['kappa_mean'] < 0.1:
                status = "GEOMETRIC"
            elif r['kappa_mean'] < 0.3:
                status = "MIXED"
            else:
                status = "MEMORIZE"
            
            flag = "PASS" if r['above_chance'] else "FAIL"
            print(f"{k:>3} | {cond:15} | "
                  f"{r['test_acc_mean']:>8.3f}+/-{r['test_acc_std']:.3f} | "
                  f"{r['kappa_mean']:>6.3f}+/-{r['kappa_std']:.3f} | "
                  f"{status:>8} {flag}")
        print("-" * 80)
    
    print("\nCEILING TEST ANALYSIS:")
    print("=" * 80)
    for k in sorted(results.keys()):
        for cond in ['max_ab', 'quad_mod_k']:
            print(f"\nk={k} {cond}:")
            print(f"  {'Mult':>6} | {'Test Acc':>10} | {'kappa':>10}")
            print(f"  {'-'*6} | {'-'*10} | {'-'*10}")
            for mult in [1, 2, 4, 8]:
                key = f"{cond}_{mult}x" if mult > 1 else cond
                r = results[k].get(key)
                if r:
                    print(f"  {mult:>6}x | {r['test_acc_mean']:>10.3f} | {r['kappa_mean']:>10.3f}")

def save_results(results, path="results/random_lut_control_v2/results.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k,v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.item()
        return obj
    
    with open(path, 'w') as f:
        json.dump(clean(results), f, indent=2)
    print(f"\nResults saved to {path}")

# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_experiment()
    print_final_report(results)
    save_results(results)
