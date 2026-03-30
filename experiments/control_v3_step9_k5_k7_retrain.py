"""
Control v3 Step 9: Retrain k=5, k=7 Fresh
==========================================
Same settings as k=11/13/17 - verify if they were also undertrained
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import math
from scipy.optimize import curve_fit
from scipy import stats
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
        popt, pcov = curve_fit(sigmoid, x_arr, y_arr, p0=[0.3, 0.1], bounds=([-0.5, 0.01], [1.0, 0.5]))
        theta, beta = popt
        y_pred = sigmoid(x_arr, theta, beta)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return float(theta), float(beta), float(r_squared), pcov
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None, None, None

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
    
    return {
        'train_acc': tr_acc,
        'test_acc': te_acc,
        'heatmap': heatmap.tolist(),
        'wrapped_distance_table': wrapped_table,
        'model': model
    }

def main():
    k_values = [5, 7]
    n_seeds = 10
    n_epochs = 150
    lr = 0.1
    hidden_mult = 1
    
    print(f"=== Retraining k={k_values} fresh ===")
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
                'wrapped_distance_table': result['wrapped_distance_table']
            })
            
            print(f"Train: {result['train_acc']:.4f}, Test: {result['test_acc']:.4f}")
        
        train_accs = [r['train_acc'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        
        print(f"  Train: {np.mean(train_accs):.4f} +/- {np.std(train_accs):.4f}")
        print(f"  Test:  {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
        
        wrapped_distances = []
        wrapped_accuracies = []
        for r in results:
            for d, vals in r['wrapped_distance_table'].items():
                wrapped_distances.append(d)
                wrapped_accuracies.append(vals['mean'])
        
        unique_ds = sorted(set(wrapped_distances))
        
        x_data = [2 * d / k for d in unique_ds]
        
        accs_per_d = {d: [] for d in unique_ds}
        for r in results:
            for d, v in r['wrapped_distance_table'].items():
                accs_per_d[d].append(v['mean'])
        
        mean_acc_per_d = [np.mean(accs_per_d[d]) for d in unique_ds]
        
        theta, beta, r_squared, pcov = fit_sigmoid(x_data, mean_acc_per_d)
        
        print(f"  Sigmoid: theta={theta:.4f}, beta={beta:.4f}, R2={r_squared:.4f}")
        
        all_results[k] = {
            'theta': theta,
            'beta': beta,
            'r_squared': r_squared,
            'train_acc_mean': float(np.mean(train_accs)),
            'test_acc_mean': float(np.mean(test_accs)),
            'wrapped_distance_points': {d: float(np.mean(accs_per_d[d])) for d in unique_ds}
        }
    
    print("\n" + "="*60)
    print("=== UPDATED 5-POINT TABLE ===")
    print("="*60)
    
    current_data = {
        5: all_results[5]['theta'],
        7: all_results[7]['theta'],
        11: 0.255,
        13: 0.318,
        17: 0.452
    }
    
    print(f"| k   | theta (fresh) | theta (old) | delta |")
    print(f"|-----|---------------|-------------|-------|")
    print(f"| 5   | {all_results[5]['theta']:.4f}        | 0.069       | {all_results[5]['theta']-0.069:+.4f} |")
    print(f"| 7   | {all_results[7]['theta']:.4f}        | 0.134       | {all_results[7]['theta']-0.134:+.4f} |")
    print(f"| 11  | 0.2554        | 0.141       | +0.114 |")
    print(f"| 13  | 0.3184        | 0.318       | 0.000  |")
    print(f"| 17  | 0.4523        | 0.452       | 0.000  |")
    
    k_vals = [5, 7, 11, 13, 17]
    theta_vals = [current_data[k] for k in k_vals]
    
    from scipy.optimize import curve_fit
    
    def power_law(k, A, alpha):
        return A * np.array(k) ** alpha
    
    popt, pcov = curve_fit(power_law, k_vals, theta_vals, p0=[0.01, 1.0])
    A, alpha = popt
    
    perr = np.sqrt(np.diag(pcov))
    A_err, alpha_err = perr
    
    y_pred = power_law(np.array(k_vals), A, alpha)
    ss_res = np.sum((np.array(theta_vals) - y_pred) ** 2)
    ss_tot = np.sum((np.array(theta_vals) - np.mean(theta_vals)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    n = len(k_vals)
    p = 2
    mse = ss_res / (n - p)
    var_pred = mse * (1/n + (np.array(k_vals) - np.mean(k_vals))**2 / np.sum((np.array(k_vals) - np.mean(k_vals))**2))
    se_fit = np.sqrt(var_pred)
    
    t_val = stats.t.ppf(0.975, n - p)
    ci_lower = y_pred - t_val * se_fit
    ci_upper = y_pred + t_val * se_fit
    
    print("\n=== POWER LAW REFIT ===")
    print(f"theta = {A:.6f} * k^{alpha:.4f}")
    print(f"R² = {r2:.4f}")
    print(f"A = {A:.6f} +/- {A_err:.6f}")
    print(f"alpha = {alpha:.4f} +/- {alpha_err:.4f}")
    
    print("\n=== 95% CONFIDENCE BAND CHECK ===")
    print(f"{'k':<4} {'theta_obs':<10} {'theta_pred':<12} {'CI_lower':<10} {'CI_upper':<10} {'in_band?':<8}")
    print("-" * 60)
    
    for i, k in enumerate(k_vals):
        obs = current_data[k]
        pred = y_pred[i]
        lower = ci_lower[i]
        upper = ci_upper[i]
        in_band = lower <= obs <= upper
        print(f"{k:<4} {obs:<10.4f} {pred:<12.4f} {lower:<10.4f} {upper:<10.4f} {'YES' if in_band else 'NO':<8}")
    
    output = {
        'k5': all_results[5],
        'k7': all_results[7],
        'power_law': {
            'A': float(A),
            'alpha': float(alpha),
            'R2': float(r2),
            'A_err': float(A_err),
            'alpha_err': float(alpha_err)
        },
        'confidence_band': {
            k: {
                'observed': current_data[k],
                'predicted': float(y_pred[i]),
                'ci_lower': float(ci_lower[i]),
                'ci_upper': float(ci_upper[i]),
                'in_band': bool(ci_lower[i] <= current_data[k] <= ci_upper[i])
            } for i, k in enumerate(k_vals)
        }
    }
    
    output_path = 'results/control_v3/step9_k5_k7_retrain.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()
