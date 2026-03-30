"""
Control v3 Step 11: Train k=19, k=21 - Find k* transition
===========================================================
Test if k=19 converges (like k=17) or fails (like k=23)
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import math
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

def generate_split(table, k, seed=0):
    rng = random.Random(seed)
    pairs = list(table.keys())
    pairs_copy = pairs.copy()
    rng.shuffle(pairs_copy)
    split = int(0.8 * len(pairs_copy))
    return list(range(split)), list(range(split, len(pairs_copy)))

def compute_heatmap_detailed(model, table, k):
    """Compute heatmap with upper/lower triangle breakdown"""
    model.eval()
    with torch.no_grad():
        heatmap = np.zeros((k, k))
        upper_correct = []
        lower_correct = []
        
        for a in range(k):
            for b in range(k):
                a_t = torch.tensor([a])
                b_t = torch.tensor([b])
                pred = model(a_t, b_t).argmax().item()
                true = max(a, b)
                correct = 1.0 if pred == true else 0.0
                heatmap[a, b] = correct
                
                if a <= b:
                    upper_correct.append(correct)
                else:
                    lower_correct.append(correct)
        
        upper_acc = np.mean(upper_correct) if upper_correct else 0
        lower_acc = np.mean(lower_correct) if lower_correct else 0
        
        return heatmap, upper_acc, lower_acc

def compute_wrapped_distance_table(heatmap, k):
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
        return None, None, None

def train_and_evaluate(table, k, seed, hidden_mult=1, n_epochs=150, lr=0.1, track_loss=False):
    set_seed(seed)
    
    train_indices, test_indices = generate_split(table, k, seed)
    
    model = ZkBundleSimpleScaled(k, hidden_mult=hidden_mult)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    pairs = list(table.keys())
    a_tr = torch.tensor([pairs[i][0] for i in train_indices], dtype=torch.long)
    b_tr = torch.tensor([pairs[i][1] for i in train_indices], dtype=torch.long)
    y_tr = torch.tensor([table[pairs[i]] for i in train_indices], dtype=torch.long)
    a_te = torch.tensor([pairs[i][0] for i in test_indices], dtype=torch.long)
    b_te = torch.tensor([pairs[i][1] for i in test_indices], dtype=torch.long)
    y_te = torch.tensor([table[pairs[i]] for i in test_indices], dtype=torch.long)
    
    loss_history = [] if track_loss else None
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(a_tr, b_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
        
        if track_loss and epoch % 10 == 0:
            with torch.no_grad():
                loss_val = criterion(model(a_tr, b_tr), y_tr).item()
                loss_history.append((epoch, loss_val))
    
    model.eval()
    with torch.no_grad():
        tr_logits = model(a_tr, b_tr)
        tr_preds = tr_logits.argmax(dim=-1)
        tr_acc = (tr_preds == y_tr).float().mean().item()
        
        te_logits = model(a_te, b_te)
        te_preds = te_logits.argmax(dim=-1)
        te_acc = (te_preds == y_te).float().mean().item()
        
        final_loss = criterion(model(a_tr, b_tr), y_tr).item()
    
    heatmap, upper_acc, lower_acc = compute_heatmap_detailed(model, table, k)
    wrapped_table = compute_wrapped_distance_table(heatmap, k)
    
    return {
        'train_acc': tr_acc,
        'test_acc': te_acc,
        'final_loss': final_loss,
        'upper_triangle_acc': upper_acc,
        'lower_triangle_acc': lower_acc,
        'heatmap': heatmap.tolist(),
        'wrapped_distance_table': wrapped_table,
        'loss_history': loss_history,
        'model': model
    }

def main():
    k_values = [19, 21]
    n_seeds = 10
    n_epochs = 150
    lr = 0.1
    hidden_mult = 1
    
    print(f"=== Training k={k_values} to find k* ===")
    print(f"Seeds: {n_seeds}, Epochs: {n_epochs}, LR: {lr}")
    
    model_dir = Path('results/control_v3/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for k in k_values:
        print(f"\n=== k={k} ===")
        table = make_max_table(k)
        results = []
        
        for seed in range(n_seeds):
            print(f"  Seed {seed}/{n_seeds-1}...", end=" ")
            
            track_loss = (seed == 0)
            result = train_and_evaluate(table, k, seed, hidden_mult=hidden_mult, 
                                         n_epochs=n_epochs, lr=lr, track_loss=track_loss)
            
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
                'final_loss': result['final_loss'],
                'upper_triangle_acc': result['upper_triangle_acc'],
                'lower_triangle_acc': result['lower_triangle_acc'],
                'wrapped_distance_table': result['wrapped_distance_table']
            })
            
            print(f"Train: {result['train_acc']:.4f}, Test: {result['test_acc']:.4f}, Loss: {result['final_loss']:.4f}")
            
            if seed == 0 and result['loss_history']:
                print(f"    Loss: {result['loss_history'][0][1]:.4f} -> {result['loss_history'][-1][1]:.4f}")
        
        train_accs = [r['train_acc'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        final_losses = [r['final_loss'] for r in results]
        upper_accs = [r['upper_triangle_acc'] for r in results]
        lower_accs = [r['lower_triangle_acc'] for r in results]
        
        print(f"  Aggregate: train={np.mean(train_accs):.4f}, test={np.mean(test_accs):.4f}")
        print(f"  Losses: {np.mean(final_losses):.4f} +/- {np.std(final_losses):.4f}")
        print(f"  Upper triangle: {np.mean(upper_accs):.4f}")
        print(f"  Lower triangle: {np.mean(lower_accs):.4f}")
        
        # Fit sigmoid
        unique_ds = sorted(set(d for r in results for d in r['wrapped_distance_table'].keys()))
        accs_per_d = {d: [] for d in unique_ds}
        for r in results:
            for d, v in r['wrapped_distance_table'].items():
                accs_per_d[d].append(v['mean'])
        
        mean_acc_per_d = [np.mean(accs_per_d[d]) for d in unique_ds]
        x_data = [2 * d / k for d in unique_ds]
        theta, beta, r_squared = fit_sigmoid(x_data, mean_acc_per_d)
        
        theta_over_k = theta / k
        
        print(f"  Sigmoid: theta={theta:.4f}, theta/k={theta_over_k:.4f}, R2={r_squared:.4f}")
        
        # Diagnosis
        mean_upper = np.mean(upper_accs)
        mean_lower = np.mean(lower_accs)
        if mean_upper > 0.9 and mean_lower < 0.5:
            print(f"  DIAGNOSIS: UPPER-TRIANGLE PARTIAL SOLUTION (like k=5)")
        elif mean_upper > 0.9 and mean_lower > 0.9:
            print(f"  DIAGNOSIS: FULL SOLUTION - CONVERGED!")
        else:
            print(f"  DIAGNOSIS: PARTIAL - upper={mean_upper:.2f}, lower={mean_lower:.2f}")
        
        all_results[k] = {
            'theta': theta,
            'beta': beta,
            'r_squared': r_squared,
            'theta_over_k': theta_over_k,
            'train_acc_mean': float(np.mean(train_accs)),
            'test_acc_mean': float(np.mean(test_accs)),
            'final_loss_mean': float(np.mean(final_losses)),
            'upper_triangle_acc': float(np.mean(upper_accs)),
            'lower_triangle_acc': float(np.mean(lower_accs)),
            'wrapped_distance_points': {d: float(np.mean(accs_per_d[d])) for d in unique_ds}
        }
    
    print("\n" + "="*70)
    print("=== k* TRANSITION ANALYSIS ===")
    print("="*70)
    
    print(f"\n| k   | theta/k | final_loss | upper_acc | lower_acc | Status |")
    print(f"|-----|---------|-------------|-----------|-----------|--------|")
    
    known = {
        17: {'theta/k': 0.0266, 'loss': 0.17, 'upper': 1.0, 'lower': 1.0, 'status': 'CONVERGED'},
        23: {'theta/k': 0.0200, 'loss': 2.26, 'upper': '?', 'lower': '?', 'status': 'STUCK'},
    }
    
    for k in [17] + k_values + [23]:
        if k == 17:
            print(f"| {k:>2} | {known[k]['theta/k']:.4f}  | {known[k]['loss']:.4f}     | {known[k]['upper']:.2f}     | {known[k]['lower']:.2f}     | {known[k]['status']} |")
        elif k == 23:
            print(f"| {k:>2} | {known[k]['theta/k']:.4f}  | {known[k]['loss']:.4f}     | ?       | ?       | {known[k]['status']}    |")
        else:
            r = all_results[k]
            status = 'CONVERGED' if r['final_loss_mean'] < 1.0 else 'STUCK'
            print(f"| {k:>2} | {r['theta_over_k']:.4f}  | {r['final_loss_mean']:.4f}     | {r['upper_triangle_acc']:.2f}     | {r['lower_triangle_acc']:.2f}     | {status} |")
    
    # Determine k*
    k19_loss = all_results[19]['final_loss_mean']
    k21_loss = all_results[21]['final_loss_mean']
    
    print(f"\n=== k* CONCLUSION ===")
    if k19_loss < 1.0 and k21_loss > 2.0:
        print(f"k* is BETWEEN 19 and 21!")
        print(f"k=19: converges (loss={k19_loss:.2f})")
        print(f"k=21: stuck (loss={k21_loss:.2f})")
    elif k19_loss > 2.0 and k21_loss > 2.0:
        print(f"k* < 19 (both stuck)")
    else:
        print(f"k* > 21 (both converge)")
    
    output_path = 'results/control_v3/step11_k19_k21.json'
    with open(output_path, 'w') as f:
        json.dump({
            'k19': all_results[19],
            'k21': all_results[21],
            'k17_reference': {'loss': 0.17, 'theta_over_k': 0.0266},
            'k23_reference': {'loss': 2.26, 'theta_over_k': 0.0200}
        }, f, indent=2)
    
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()
