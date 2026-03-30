"""
Phase Resolution: k=13,17 Retrain with Wrapped Normalization
=============================================================
Testing whether theta flattens to a universal constant ~0.13
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

def train_and_get_wrapped_table(table, k, train_indices, test_indices, seed, hidden_mult=1, n_epochs=150, lr=0.1):
    set_seed(seed)
    
    model = ZkBundleSimpleScaled(k, hidden_mult=hidden_mult)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    (a_tr, b_tr, y_tr), (a_te, b_te, y_te) = make_dataset_fixed(table, k, train_indices, test_indices)
    
    for epoch in range(n_epochs):
        if epoch % 50 == 0:
            print(f"      Epoch {epoch}/{n_epochs}")
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
        wrapped_table = compute_wrapped_distance_table(heatmap, k)
    
    return {
        'test_acc': te_acc,
        'heatmap': heatmap,
        'wrapped_table': wrapped_table
    }

def main():
    K_VALUES = [13, 17]
    N_SEEDS = 10
    HIDDEN_MULT = 1
    N_EPOCHS = 150
    
    Path("results/control_v3").mkdir(parents=True, exist_ok=True)
    
    # Load existing corrected results for k=5,7,11
    with open("results/control_v3/step6_k5_7_11_corrected.json", 'r') as f:
        corrected_5_7_11 = json.load(f)
    
    new_results = {}
    
    print("="*70)
    print("RETRAINING k=13,17 with WRAPPED NORMALIZATION")
    print("="*70)
    
    for k in K_VALUES:
        print(f"\n{'='*70}")
        print(f"TRAINING k = {k}")
        print(f"{'='*70}")
        
        table = make_max_table(k)
        train_indices, test_indices = generate_split(table, k, seed=0)
        
        all_wrapped = {d: [] for d in range(k // 2 + 1)}
        
        for seed in range(N_SEEDS):
            print(f"\n  Seed {seed}:")
            result = train_and_get_wrapped_table(table, k, train_indices, test_indices, seed, 
                                                 hidden_mult=HIDDEN_MULT, n_epochs=N_EPOCHS, lr=0.1)
            
            wrapped_table = result['wrapped_table']
            for d in range(k // 2 + 1):
                if d in wrapped_table:
                    all_wrapped[d].append(wrapped_table[d]['mean'])
        
        # Aggregate across seeds
        print(f"\n--- Wrapped Distance Table for k={k} ---")
        print(f"d   wrapped_d   x=2d/k   acc")
        print("-" * 35)
        
        x_data = []
        y_data = []
        
        for d in range(k // 2 + 1):
            if all_wrapped[d]:
                mean_acc = np.mean(all_wrapped[d])
                x = 2 * d / k
                x_data.append(x)
                y_data.append(mean_acc)
                diag_mark = "(diag)" if d == 0 else ""
                print(f"{d:2d}  {d:10d}  {x:6.3f}   {mean_acc:.3f}  {diag_mark}")
        
        theta, beta, r2 = fit_sigmoid(x_data, y_data)
        
        print(f"\nFitted: theta={theta:.4f}, beta={beta:.4f}, R2={r2:.4f}")
        
        new_results[k] = {
            'theta': theta,
            'beta': beta,
            'r_squared': r2,
            'wrapped_table': {d: float(np.mean(all_wrapped[d])) for d in all_wrapped if all_wrapped[d]}
        }
    
    # Full comparison table
    print("\n" + "="*70)
    print("FULL 5-POINT BEFORE/AFTER TABLE")
    print("="*70)
    
    # BEFORE: old buggy normalization
    old_theta = [0.0527, 0.1248, 0.1548, 0.3184, 0.4523]
    old_beta = [0.1173, 0.1121, 0.1038, 0.1682, 0.2637]
    
    # AFTER: corrected for k=5,7,11 + new for k=13,17
    new_theta = [
        corrected_5_7_11['5']['theta'],
        corrected_5_7_11['7']['theta'],
        corrected_5_7_11['11']['theta'],
        new_results[13]['theta'],
        new_results[17]['theta']
    ]
    new_beta = [
        corrected_5_7_11['5']['beta'],
        corrected_5_7_11['7']['beta'],
        corrected_5_7_11['11']['beta'],
        new_results[13]['beta'],
        new_results[17]['beta']
    ]
    new_r2 = [
        corrected_5_7_11['5']['r_squared'],
        corrected_5_7_11['7']['r_squared'],
        corrected_5_7_11['11']['r_squared'],
        new_results[13]['r_squared'],
        new_results[17]['r_squared']
    ]
    
    print(f"\n{'k':>4}  {'theta':>8}  {'beta':>8}  {'R2':>7}  |  {'theta':>8}  {'beta':>8}  {'R2':>7}")
    print(f"{'':>4}  {'BEFORE':>8}  {'BEFORE':>8}  {'BEFORE':>7}  |  {'AFTER':>8}  {'AFTER':>8}  {'AFTER':>7}")
    print("-" * 65)
    
    k_list = [5, 7, 11, 13, 17]
    for i, k in enumerate(k_list):
        print(f"{k:>4}  {old_theta[i]:>8.4f}  {old_beta[i]:>8.4f}  {0.0:>7.3f}  |  {new_theta[i]:>8.4f}  {new_beta[i]:>8.4f}  {new_r2[i]:>7.3f}")
    
    print("\n" + "="*70)
    print("SCALING LAW COMPETITION (corrected)")
    print("="*70)
    
    k_arr = np.array(k_list, dtype=float)
    theta_arr = np.array(new_theta, dtype=float)
    beta_arr = np.array(new_beta, dtype=float)
    
    # Fit models
    def model_log(k, A):
        return A * np.log(k)
    popt_log, _ = curve_fit(model_log, k_arr, theta_arr, p0=[0.1])
    A_log = popt_log.item()
    pred_log = model_log(k_arr, A_log)
    ss_res_log = np.sum((theta_arr - pred_log)**2)
    ss_tot = np.sum((theta_arr - np.mean(theta_arr))**2)
    r2_log = 1 - ss_res_log/ss_tot
    
    def model_power(k, A, alpha):
        return A * np.power(k, alpha)
    popt_power, _ = curve_fit(model_power, k_arr, theta_arr, p0=[1, -0.5])
    A_power = popt_power[0].item()
    alpha_power = popt_power[1].item()
    pred_power = model_power(k_arr, A_power, alpha_power)
    ss_res_power = np.sum((theta_arr - pred_power)**2)
    r2_power = 1 - ss_res_power/ss_tot
    
    def model_constant(k, C):
        return np.full_like(k, C, dtype=float)
    popt_const, _ = curve_fit(model_constant, k_arr, theta_arr, p0=[0.13])
    C_const = popt_const.item()
    pred_const = model_constant(k_arr, C_const)
    ss_res_const = np.sum((theta_arr - pred_const)**2)
    r2_const = 1 - ss_res_const/ss_tot
    
    def model_log_plus(k, A, B):
        return A * np.log(k) + B
    popt_lp, _ = curve_fit(model_log_plus, k_arr, theta_arr, p0=[0.1, 0])
    A_lp = popt_lp[0].item()
    B_lp = popt_lp[1].item()
    pred_lp = model_log_plus(k_arr, A_lp, B_lp)
    ss_res_lp = np.sum((theta_arr - pred_lp)**2)
    r2_lp = 1 - ss_res_lp/ss_tot
    
    print(f"\n{'Model':>12}  {'R2':>8}  {'pred_k23':>10}")
    print("-" * 35)
    print(f"{'log(k)':>12}  {r2_log:>8.4f}  {model_log(np.array([23]), A_log).item():>10.4f}")
    print(f"{'k^alpha':>12}  {r2_power:>8.4f}  {model_power(np.array([23]), A_power, alpha_power).item():>10.4f}")
    print(f"{'CONSTANT':>12}  {r2_const:>8.4f}  {C_const:>10.4f}")
    print(f"{'log(k)+B':>12}  {r2_lp:>8.4f}  {model_log_plus(np.array([23]), A_lp, B_lp).item():>10.4f}")
    
    # Find winner
    models = {'log(k)': r2_log, 'k^alpha': r2_power, 'CONSTANT': r2_const, 'log(k)+B': r2_lp}
    winner = max(models, key=models.get)
    print(f"\n*** WINNING MODEL: {winner} ***")
    
    # Beta consistency
    beta_mean = np.mean(beta_arr)
    beta_std = np.std(beta_arr)
    print(f"\n--- Beta Consistency ---")
    print(f"Beta values: {[f'{b:.4f}' for b in beta_arr]}")
    print(f"Mean: {beta_mean:.4f}, Std: {beta_std:.4f}")
    if beta_std < 0.015:
        print("Verdict: CONFIRMED (beta is universal)")
    else:
        print("Verdict: DRIFTING (beta has k-dependence)")
    
    # Save final JSON
    final_results = {
        "normalization": "wrapped_2d_over_k",
        "normalization_note": "x = 2*min(d,k-d)/k, range [0,1]",
        "previous_normalization_bug": "k=5,7,11 used d/k; k=13,17 used 2d/k - now fixed",
        "k_results": {
            str(k): {
                'theta': new_theta[i],
                'beta': new_beta[i],
                'r_squared': new_r2[i]
            } for i, k in enumerate(k_list)
        },
        "scaling_law": {
            "k_values": k_list,
            "theta_values": new_theta,
            "beta_values": new_beta,
            "fits": {
                "log_k": {"A": A_log, "R2": r2_log, "pred_k23": float(model_log(np.array([23]), A_log).item())},
                "power": {"A": A_power, "alpha": alpha_power, "R2": r2_power, "pred_k23": float(model_power(np.array([23]), A_power, alpha_power).item())},
                "constant": {"C": C_const, "R2": r2_const},
                "log_plus": {"A": A_lp, "B": B_lp, "R2": r2_lp}
            },
            "winning_model": winner,
            "beta_universal": {
                "mean": float(beta_mean),
                "std": float(beta_std),
                "verdict": "CONFIRMED" if beta_std < 0.015 else "DRIFTING"
            }
        }
    }
    
    with open("results/control_v3/phase_resolution_fit.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nSaved: results/control_v3/phase_resolution_fit.json")
    
    # Generate plot
    print("\nGenerating plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Accuracy vs wrapped distance
    ax1 = axes[0]
    colors = {5: 'blue', 7: 'green', 11: 'red', 13: 'orange', 17: 'purple'}
    x_fit = np.linspace(0, 1, 100)
    
    for i, k in enumerate(k_list):
        # Get data points
        if k in [5, 7, 11]:
            wrapped = corrected_5_7_11[str(k)].get('distance_table', {})
        else:
            wrapped = new_results[k]['wrapped_table']
        
        x_pts = [2*int(d)/k for d in wrapped.keys()]
        y_pts = list(wrapped.values())
        ax1.scatter(x_pts, y_pts, color=colors[k], s=60, label=f'k={k}')
        
        y_fit = sigmoid(x_fit, new_theta[i], new_beta[i])
        ax1.plot(x_fit, y_fit, color=colors[k], linestyle='--', alpha=0.5)
    
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax1.set_xlabel('x = 2*min(d,k-d)/k')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Phase Resolution (Wrapped)')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: theta vs k
    ax2 = axes[1]
    ax2.scatter(k_list, new_theta, color='black', s=80, zorder=5)
    
    k_fit = np.linspace(5, 25, 100)
    ax2.plot(k_fit, model_log(k_fit, A_log), 'b--', alpha=0.6, label=f'log(k) R2={r2_log:.3f}')
    ax2.plot(k_fit, model_power(k_fit, A_power, alpha_power), 'r--', alpha=0.6, label=f'k^alpha R2={r2_power:.3f}')
    ax2.plot(k_fit, np.full_like(k_fit, C_const), 'g--', alpha=0.6, label=f'CONST R2={r2_const:.3f}')
    ax2.axhline(y=C_const, color='green', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('k')
    ax2.set_ylabel('theta')
    ax2.set_title(f'Theta(k): Winner = {winner}')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: beta vs k
    ax3 = axes[2]
    ax3.scatter(k_list, new_beta, color='black', s=80, zorder=5)
    ax3.axhline(y=beta_mean, color='red', linestyle='--', alpha=0.7, label=f'mean={beta_mean:.3f}')
    ax3.set_xlabel('k')
    ax3.set_ylabel('beta')
    ax3.set_title(f'Beta(k): Std={beta_std:.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/control_v3/phase_resolution_scaling.png", dpi=150, bbox_inches='tight')
    print("Saved: results/control_v3/phase_resolution_scaling.png")
    
    return final_results

if __name__ == "__main__":
    results = main()
