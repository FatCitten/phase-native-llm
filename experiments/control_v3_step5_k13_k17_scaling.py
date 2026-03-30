"""
Phase Resolution Scaling: k=13 and k=17 Extension
==================================================
Train models, evaluate distance-from-diagonal, fit sigmoids,
identify scaling law, and update plots.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    distance_acc = {d: [] for d in range(k // 2 + 1)}
    
    for a in range(k):
        for b in range(k):
            d_raw = abs(a - b)
            d = min(d_raw, k - d_raw)
            distance_acc[d].append(heatmap[a, b])
    
    result = {}
    max_d = k // 2 + 1
    for d in range(max_d):
        if distance_acc[d]:
            result[d] = {
                'mean': float(np.mean(distance_acc[d])),
                'count': len(distance_acc[d])
            }
    return result

def sigmoid(x, theta, beta):
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-(x - theta) / beta))

def fit_sigmoid(x_data, y_data):
    x_arr = np.asarray(x_data)
    y_arr = np.asarray(y_data)
    
    try:
        popt, _ = curve_fit(sigmoid, x_arr, y_arr, p0=[0.3, 0.05], bounds=([0, 0.001], [0.5, 0.5]))
        theta, beta = popt
        
        y_pred = sigmoid(x_arr, theta, beta)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return theta, beta, r_squared
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None, None

def train_and_get_heatmap(table, k, train_indices, test_indices, seed, hidden_mult=1, n_epochs=150, lr=0.1):
    set_seed(seed)
    
    model = ZkBundleSimpleScaled(k, hidden_mult=hidden_mult)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    (a_tr, b_tr, y_tr), (a_te, b_te, y_te) = make_dataset_fixed(table, k, train_indices, test_indices)
    
    for epoch in range(n_epochs):
        if epoch % 30 == 0:
            print(f"    Epoch {epoch}/{n_epochs}")
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

def run_k_extension():
    NEW_K_VALUES = [13, 17]
    N_SEEDS = 10
    HIDDEN_MULT = 1
    N_EPOCHS = 150
    
    Path("results/control_v3").mkdir(parents=True, exist_ok=True)
    
    new_results = {}
    
    for k in NEW_K_VALUES:
        print(f"\n{'='*70}")
        print(f"TRAINING k = {k}")
        print(f"{'='*70}")
        
        table = make_max_table(k)
        train_indices, test_indices = generate_split(table, k, seed=0)
        
        all_distance = {d: [] for d in range(k)}
        all_test_acc = []
        
        for seed in range(N_SEEDS):
            print(f"\n  Seed {seed}:")
            result = train_and_get_heatmap(table, k, train_indices, test_indices, seed, 
                                          hidden_mult=HIDDEN_MULT, n_epochs=N_EPOCHS, lr=0.1)
            
            heatmap = result['heatmap']
            all_test_acc.append(result['test_acc'])
            
            distance_table = compute_distance_table(heatmap, k)
            for d in range(k):
                if d in distance_table:
                    all_distance[d].append(distance_table[d]['mean'])
        
        avg_test_acc = np.mean(all_test_acc)
        
        print(f"\n--- Distance-from-Diagonal Table for k={k} ---")
        print(f"d = min(|a-b|, k-|a-b|)  (count):  accuracy")
        print("-" * 35)
        
        distance_means = {}
        max_d = k // 2 + 1
        for d in range(max_d):
            if all_distance[d]:
                mean_acc = np.mean(all_distance[d])
                count = len(all_distance[d])
                distance_means[d] = mean_acc
                diag_marker = "(diagonal)" if d == 0 else ""
                print(f"d={d:2d}  (n={count:2d}):  {mean_acc:.3f}  {diag_marker}")
        
        # Fit sigmoid - use d/(k/2) as normalized x for wrapped distance
        x_data = [2 * d / k for d in distance_means.keys()]  # d_eff/k_eff where k_eff = k/2
        y_data = list(distance_means.values())
        
        theta, beta, r2 = fit_sigmoid(x_data, y_data)
        
        print(f"\n--- Sigmoid Fit for k={k} ---")
        print(f"theta = {theta:.4f}, beta = {beta:.4f}, R2 = {r2:.4f}")
        
        if r2 < 0.90:
            print(f"WARNING: R2 < 0.90 - fit may be unreliable")
        
        new_results[k] = {
            'theta': theta,
            'beta': beta,
            'r_squared': r2,
            'test_acc': float(avg_test_acc),
            'distance_table': distance_means
        }
        
        # Save model
        print(f"Saving results for k={k}...")
    
    return new_results

def scaling_analysis(existing_results, new_results):
    print("\n" + "="*70)
    print("SCALING LAW IDENTIFICATION")
    print("="*70)
    
    k_values = [5, 7, 11, 13, 17]
    theta_values = [
        existing_results['per_k']['5']['theta'],
        existing_results['per_k']['7']['theta'],
        existing_results['per_k']['11']['theta'],
        new_results[13]['theta'],
        new_results[17]['theta']
    ]
    beta_values = [
        existing_results['per_k']['5']['beta'],
        existing_results['per_k']['7']['beta'],
        existing_results['per_k']['11']['beta'],
        new_results[13]['beta'],
        new_results[17]['beta']
    ]
    
    print("\nk values:", k_values)
    print("theta values:", [f"{t:.4f}" for t in theta_values])
    print("beta values:", [f"{b:.4f}" for b in beta_values])
    
    # Fit candidate scaling laws
    k_arr = np.array(k_values, dtype=float)
    theta_arr = np.array(theta_values, dtype=float)
    
    def fit_and_score(predict_fn, params, k_pred=None):
        theta_pred = predict_fn(k_arr, *params)
        ss_res = np.sum((theta_arr - theta_pred) ** 2)
        ss_tot = np.sum((theta_arr - np.mean(theta_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        pred_k23 = predict_fn(np.array([23]), *params)[0] if k_pred is None else None
        return r2, theta_pred, pred_k23
    
    # (A) theta = A * log(k)
    def model_log(k, A):
        return A * np.log(k)
    popt_log, _ = curve_fit(model_log, k_arr, theta_arr, p0=[0.1])
    A_log = popt_log.item()
    r2_log, _, pred23_log = fit_and_score(model_log, [A_log])
    
    # (B) theta = A / sqrt(k)
    def model_inv_sqrt(k, A):
        return A / np.sqrt(k)
    popt_sqrt, _ = curve_fit(model_inv_sqrt, k_arr, theta_arr, p0=[0.5])
    A_sqrt = popt_sqrt.item()
    r2_sqrt, _, pred23_sqrt = fit_and_score(model_inv_sqrt, [A_sqrt])
    
    # (C) theta = A * k^alpha
    def model_power(k, A, alpha):
        return A * np.power(k, alpha)
    popt_power, _ = curve_fit(model_power, k_arr, theta_arr, p0=[1, -0.5])
    A_power = popt_power[0].item()
    alpha = popt_power[1].item()
    r2_power, _, pred23_power = fit_and_score(model_power, [A_power, alpha])
    
    # (D) theta = A * log(k) + B
    def model_log_plus(k, A, B):
        return A * np.log(k) + B
    popt_log_plus, _ = curve_fit(model_log_plus, k_arr, theta_arr, p0=[0.1, 0.0])
    A_lp = popt_log_plus[0].item()
    B_lp = popt_log_plus[1].item()
    r2_lp, _, pred23_lp = fit_and_score(model_log_plus, [A_lp, B_lp])
    
    print("\n--- Scaling Law Comparison ---")
    print(f"{'Model':<12} {'R2':>8} {'pred_k23':>10}")
    print("-" * 35)
    pred23_log = float(model_log(np.array([23]), A_log)[0])
    pred23_sqrt = float(model_inv_sqrt(np.array([23]), A_sqrt)[0])
    pred23_power = float(model_power(np.array([23]), A_power, alpha)[0])
    pred23_lp = float(model_log_plus(np.array([23]), A_lp, B_lp)[0])
    print(f"{'log(k)':<12} {r2_log:>8.4f} {pred23_log:>10.4f}")
    print(f"{'1/sqrt(k)':<12} {r2_sqrt:>8.4f} {pred23_sqrt:>10.4f}")
    print(f"{'k^alpha':<12} {r2_power:>8.4f} {pred23_power:>10.4f}")
    print(f"{'log(k)+B':<12} {r2_lp:>8.4f} {pred23_lp:>10.4f}")
    
    # Find winning model
    r2_scores = {'log_k': r2_log, 'inv_sqrt': r2_sqrt, 'power': r2_power, 'log_plus': r2_lp}
    winning = max(r2_scores, key=r2_scores.get)
    print(f"\nWinning model: {winning} (R2 = {r2_scores[winning]:.4f})")
    
    # Beta consistency
    beta_mean = np.mean(beta_values)
    beta_std = np.std(beta_values)
    print(f"\n--- Beta Consistency ---")
    print(f"Beta values: {[f'{b:.4f}' for b in beta_values]}")
    print(f"Mean: {beta_mean:.4f}, Std: {beta_std:.4f}")
    if beta_std < 0.01:
        beta_verdict = "CONFIRMED"
        print("Verdict: UNIVERSAL CONSTANT (std < 0.01)")
    else:
        beta_verdict = "DRIFTING"
        print("Verdict: beta is drifting with k")
    
    return {
        'k_values': k_values,
        'theta_values': theta_values,
        'beta_values': beta_values,
        'fits': {
            'log_k': {'A': float(A_log), 'R2': float(r2_log), 'pred_k23': float(model_log(23, A_log))},
            'inv_sqrt': {'A': float(A_sqrt), 'R2': float(r2_sqrt), 'pred_k23': float(model_inv_sqrt(23, A_sqrt))},
            'power': {'A': float(A_power), 'alpha': float(alpha), 'R2': float(r2_power), 'pred_k23': float(model_power(23, A_power, alpha))},
            'log_plus': {'A': float(A_lp), 'B': float(B_lp), 'R2': float(r2_lp), 'pred_k23': float(model_log_plus(23, A_lp, B_lp))}
        },
        'winning_model': winning,
        'beta_universal': {
            'mean': float(beta_mean),
            'std': float(beta_std),
            'verdict': beta_verdict
        }
    }

def generate_plots(existing_results, new_results, scaling_info):
    print("\n" + "="*70)
    print("GENERATING UPDATED PLOTS")
    print("="*70)
    
    k_values = scaling_info['k_values']
    colors = {5: 'blue', 7: 'green', 11: 'red', 13: 'orange', 17: 'purple'}
    
    # Collect all distance data
    all_data = {}
    # Existing data
    existing_data = {
        5: {0: 0.400, 1: 0.750, 2: 1.000, 3: 1.000, 4: 1.000},
        7: {0: 0.286, 1: 0.500, 2: 0.800, 3: 1.000, 4: 1.000, 5: 1.000, 6: 1.000},
        11: {0: 0.182, 1: 0.400, 2: 0.556, 3: 0.625, 4: 1.000, 5: 1.000, 6: 1.000, 7: 1.000, 8: 1.000, 9: 1.000, 10: 1.000}
    }
    for k in [5, 7, 11]:
        all_data[k] = {'d_over_k': [], 'acc': []}
        for d, acc in existing_data[k].items():
            all_data[k]['d_over_k'].append(d / k)
            all_data[k]['acc'].append(acc)
    # New data
    for k in [13, 17]:
        all_data[k] = {'d_over_k': [], 'acc': []}
        dt = new_results[k]['distance_table']
        for d, acc in dt.items():
            all_data[k]['d_over_k'].append(d / k)
            all_data[k]['acc'].append(acc)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs d/k
    ax1 = axes[0]
    x_fit = np.linspace(0, 0.5, 100)
    
    for k in k_values:
        x = all_data[k]['d_over_k']
        y = all_data[k]['acc']
        
        ax1.scatter(x, y, color=colors[k], s=60, zorder=5, label=f'k={k}')
        
        theta = existing_results['per_k'][str(k)]['theta'] if k in [5, 7, 11] else new_results[k]['theta']
        beta = existing_results['per_k'][str(k)]['beta'] if k in [5, 7, 11] else new_results[k]['beta']
        
        y_fit = sigmoid(x_fit, theta, beta)
        ax1.plot(x_fit, y_fit, color=colors[k], linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax1.set_xlabel('d/k (normalized distance from diagonal)', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Phase Resolution: Accuracy vs Normalized Distance', fontsize=12)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: theta vs k with fits
    ax2 = axes[1]
    k_arr = np.array(k_values)
    theta_arr = np.array(scaling_info['theta_values'])
    
    ax2.scatter(k_arr, theta_arr, color='black', s=80, zorder=5, label='Observed theta')
    
    k_fit = np.linspace(5, 25, 100)
    
    # Plot each fit
    fits = scaling_info['fits']
    ax2.plot(k_fit, fits['log_k']['A'] * np.log(k_fit), 'b--', alpha=0.6, label=f"log(k) R2={fits['log_k']['R2']:.3f}")
    ax2.plot(k_fit, fits['inv_sqrt']['A'] / np.sqrt(k_fit), 'g--', alpha=0.6, label=f"1/sqrt(k) R2={fits['inv_sqrt']['R2']:.3f}")
    ax2.plot(k_fit, fits['power']['A'] * np.power(k_fit, fits['power']['alpha']), 'r--', alpha=0.6, label=f"k^alpha R2={fits['power']['R2']:.3f}")
    ax2.plot(k_fit, fits['log_plus']['A'] * np.log(k_fit) + fits['log_plus']['B'], 'm--', alpha=0.6, label=f"log(k)+B R2={fits['log_plus']['R2']:.3f}")
    
    ax2.axvline(x=23, color='gray', linestyle=':', alpha=0.5)
    ax2.scatter([23], [fits['power']['pred_k23']], color='red', marker='*', s=150, zorder=6, label=f'pred k=23')
    
    ax2.set_xlabel('k', fontsize=11)
    ax2.set_ylabel('theta (threshold)', fontsize=11)
    ax2.set_title(f'Scaling Law: theta(k)\nWinning: {scaling_info["winning_model"]}', fontsize=12)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(3, 27)
    
    plt.tight_layout()
    plt.savefig("results/control_v3/phase_resolution_scaling.png", dpi=150, bbox_inches='tight')
    print("Saved: results/control_v3/phase_resolution_scaling.png")

def main():
    # Load existing results
    with open("results/control_v3/phase_resolution_fit.json", 'r') as f:
        existing_results = json.load(f)
    
    print("="*70)
    print("PHASE RESOLUTION SCALING: k=13 and k=17 EXTENSION")
    print("="*70)
    
    # Train new k values
    new_results = run_k_extension()
    
    # Scaling law analysis
    scaling_info = scaling_analysis(existing_results, new_results)
    
    # Generate plots
    generate_plots(existing_results, new_results, scaling_info)
    
    # Update JSON
    existing_results['per_k']['13'] = {
        'theta': new_results[13]['theta'],
        'beta': new_results[13]['beta'],
        'r_squared': new_results[13]['r_squared']
    }
    existing_results['per_k']['17'] = {
        'theta': new_results[17]['theta'],
        'beta': new_results[17]['beta'],
        'r_squared': new_results[17]['r_squared']
    }
    existing_results['scaling_law'] = scaling_info
    
    with open("results/control_v3/phase_resolution_fit.json", 'w') as f:
        json.dump(existing_results, f, indent=2)
    print("\nSaved: results/control_v3/phase_resolution_fit.json")
    
    return existing_results

if __name__ == "__main__":
    results = main()
