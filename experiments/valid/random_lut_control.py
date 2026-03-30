"""
THREE-CONDITION CONTROL EXPERIMENT
===================================
Tests whether ZkBundle requires GROUP STRUCTURE (Hypothesis X)
or just PHASE-ENCODABLE PATTERNS (Hypothesis Y).

Condition 1: Z_k addition      - group structure       - expect 100%, HIGH κ
Condition 2: Random LUT        - no structure          - expect 1/k,  LOW κ  
Condition 3: max(a,b) mod k    - structure, not group  - THE DISCRIMINATOR

Run: python experiments/valid/random_lut_control.py
Results saved to: results/random_lut_control/
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
    return {(a,b): (a+b) % k 
            for a,b in iproduct(range(k), range(k))}

def make_random_lut(k, seed=42):
    """Completely random mapping — no structure whatsoever
    
    FIXED seed=42 for ALL k, ALL experiment seeds.
    This is intentional: we want ONE fixed adversarial table.
    If architecture fails this, it's definitely geometric.
    If it passes, we have a serious problem to investigate.
    """
    rng = random.Random(seed)
    return {(a,b): rng.randint(0, k-1) 
            for a,b in iproduct(range(k), range(k))}

def make_max_table(k):
    """max(a,b) mod k — deterministic structure, NOT a group
    
    Properties:
    - Commutative: max(a,b) = max(b,a) 
    - NOT associative in general mod k
    - No identity element
    - No inverses
    - MLP can fit this perfectly
    - Phase architecture: unknown 
    """
    return {(a,b): max(a,b) % k 
            for a,b in iproduct(range(k), range(k))}

# ─── ORIGINAL PROVEN ARCHITECTURE ───────────────────────────────────────────

class ZkBundle(nn.Module):
    """
    Original proven phase-native architecture.
    Uses direct phase addition - this is what achieves 100% on Z_k addition.
    """
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
        """
        Compute phase coherence κ ∈ [0,1].
        
        κ = |mean(exp(i·θ))| across the k input phases
        
        HIGH κ → phases clustered at geometric positions → structure found
        LOW κ  → phases scattered → no geometric solution
        """
        with torch.no_grad():
            phases = self.input_phases.data  # (k,)
            # Complex mean
            mean_real = torch.cos(phases).mean()
            mean_imag = torch.sin(phases).mean()
            kappa = (mean_real**2 + mean_imag**2).sqrt().item()
            return kappa

# ─── DATASET ─────────────────────────────────────────────────────────────────

def make_dataset(table, k):
    """
    Convert lookup table to (inputs, labels) tensors.
    Uses FULL table (all k² pairs).
    
    Train/test split: 80/20.
    """
    pairs = list(table.keys())
    random.shuffle(pairs)
    
    split = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    test_pairs  = pairs[split:]
    
    def to_tensors(ps):
        a = torch.tensor([p[0] for p in ps], dtype=torch.long)
        b = torch.tensor([p[1] for p in ps], dtype=torch.long)
        y = torch.tensor([table[p] for p in ps], dtype=torch.long)
        return a, b, y
    
    return to_tensors(train_pairs), to_tensors(test_pairs)

# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_one_run(table, k, seed, n_epochs=150, lr=0.1):
    """
    Single training run with proven hyperparameters.
    """
    set_seed(seed)
    
    model = ZkBundle(k)
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
    
    # ── Evaluation ──
    model.eval()
    with torch.no_grad():
        tr_logits = model(a_tr, b_tr)
        tr_acc = (tr_logits.argmax(dim=-1) == y_tr).float().mean().item()
        
        te_logits = model(a_te, b_te)
        te_acc = (te_logits.argmax(dim=-1) == y_te).float().mean().item()
        
        kappa = model.get_kappa()
    
    return {
        'train_acc': tr_acc,
        'test_acc':  te_acc,
        'kappa':     kappa,
        'chance_level': 1.0 / k,
    }

# ─── MAIN EXPERIMENT LOOP ────────────────────────────────────────────────────

def run_experiment():
    
    K_VALUES  = [3, 5, 7, 11]
    N_SEEDS   = 10
    RESULTS   = {}
    
    CONDITIONS = {
        'addition': make_addition_table,
        'random_lut': lambda k: make_random_lut(k, seed=42),
        'max_mod_k': make_max_table,
    }
    
    print("=" * 60)
    print("THREE-CONDITION CONTROL EXPERIMENT")
    print("=" * 60)
    print(f"k values:   {K_VALUES}")
    print(f"Seeds:      {N_SEEDS} per point")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Total runs: {len(K_VALUES) * len(CONDITIONS) * N_SEEDS}")
    print("=" * 60)
    
    for k in K_VALUES:
        RESULTS[k] = {}
        chance = 1.0 / k
        
        for cond_name, table_fn in CONDITIONS.items():
            table = table_fn(k)
            seed_results = []
            
            print(f"\nk={k:2d} | {cond_name:12s} | chance={chance:.3f}")
            print(f"       seed: ", end="", flush=True)
            
            for seed in range(N_SEEDS):
                print(f"{seed}", end=" ", flush=True)
                result = train_one_run(table, k, seed=seed)
                seed_results.append(result)
            
            # Aggregate across seeds
            test_accs = [r['test_acc'] for r in seed_results]
            kappas    = [r['kappa']    for r in seed_results]
            
            summary = {
                'test_acc_mean':  float(np.mean(test_accs)),
                'test_acc_std':   float(np.std(test_accs)),
                'kappa_mean':     float(np.mean(kappas)),
                'kappa_std':      float(np.std(kappas)),
                'chance_level':   chance,
                'above_chance':   float(np.mean(test_accs)) > chance * 1.5,
                'raw':            seed_results,
            }
            
            RESULTS[k][cond_name] = summary
            
            print(f"\n       acc={summary['test_acc_mean']:.3f}+/-{summary['test_acc_std']:.3f}"
                  f"  kappa={summary['kappa_mean']:.3f}+/-{summary['kappa_std']:.3f}"
                  f"  {'PASS' if summary['above_chance'] else 'FAIL'}")
    
    return RESULTS

# ─── REPORTING ───────────────────────────────────────────────────────────────

def print_final_report(results):
    print("\n")
    print("=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"{'k':>3} | {'Condition':15} | {'Test Acc':>10} | {'kappa':>8} | {'vs Chance':>12}")
    print("-" * 70)
    
    for k in sorted(results.keys()):
        for cond in ['addition', 'random_lut', 'max_mod_k']:
            r = results[k][cond]
            chance_str = f"(chance={r['chance_level']:.2f})"
            flag = "PASS" if r['above_chance'] else "FAIL"
            print(f"{k:>3} | {cond:15} | "
                  f"{r['test_acc_mean']:>8.3f}+/-{r['test_acc_std']:.3f} | "
                  f"{r['kappa_mean']:>6.3f}+/-{r['kappa_std']:.3f} | "
                  f"{flag} {chance_str}")
        print("-" * 70)
    
    # Hypothesis verdict
    print("\nHYPOTHESIS VERDICT:")
    print("-" * 70)
    
    for k in sorted(results.keys()):
        rnd_acc = results[k]['random_lut']['test_acc_mean']
        max_acc = results[k]['max_mod_k']['test_acc_mean']
        chance  = 1.0 / k
        
        rnd_at_chance = rnd_acc < chance * 1.5
        max_above     = max_acc > chance * 1.5
        
        if rnd_at_chance and max_above:
            verdict = "OUTCOME B: phase encodes non-group structure"
        elif rnd_at_chance and not max_above:
            verdict = "OUTCOME A: group structure REQUIRED - thesis supported"
        elif not rnd_at_chance:
            verdict = "OUTCOME C: BYPASS DETECTED - architecture audit needed"
        else:
            verdict = "AMBIGUOUS"
        
        print(f"k={k}: {verdict}")

def save_results(results, path="results/random_lut_control/results.json"):
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
