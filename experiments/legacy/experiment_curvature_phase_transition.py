"""
Experiment 3: Curvature Phase Transition
=========================================
For k=7, train at different noise levels sigma
Compute Q = mean(|off-diagonal|) - mean(|diagonal|)
Expect sharp drop at sigma* ≈ 0.26
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


class ZkBundle(nn.Module):
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
    
    def forward_with_noise(self, x1, x2, noise_sigma):
        p1_base = self.input_phases[x1]
        p2_base = self.input_phases[x2]
        p1 = p1_base + torch.randn_like(p1_base) * noise_sigma
        p2 = p2_base + torch.randn_like(p2_base) * noise_sigma
        phi = (p1 + p2) % (2 * math.pi)
        dists = torch.abs(phi.unsqueeze(-1) - self.output_phases.unsqueeze(0))
        dists = dists % (2 * math.pi)
        dists = torch.min(dists, 2 * math.pi - dists)
        return -dists
    
    def get_phases(self):
        return self.input_phases.detach().cpu().numpy()
    
    def get_output_phases(self):
        return self.output_phases.detach().cpu().numpy()


def generate_zk_data(k, n_samples):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    return x1, x2, y


def train_zk_with_noise(k, n_samples=1000, epochs=150, seed=42, noise_sigma=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1, x2, y = generate_zk_data(k, n_samples)
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        if noise_sigma > 0:
            outputs = model.forward_with_noise(x1, x2, noise_sigma)
        else:
            outputs = model(x1, x2)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model


def compute_Q(curvature_matrix):
    """Q = mean(|off-diagonal|) - mean(|diagonal|)"""
    k = curvature_matrix.shape[0]
    diagonal = np.array([curvature_matrix[i, i] for i in range(k)])
    off_diagonal = curvature_matrix[~np.eye(k, dtype=bool)]
    
    Q = np.mean(np.abs(off_diagonal)) - np.mean(np.abs(diagonal))
    return Q, np.mean(np.abs(off_diagonal)), np.mean(np.abs(diagonal))


def compute_curvature_matrix(model, k):
    output_phases = model.get_output_phases()
    
    curvature = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            diff = abs(output_phases[i] - output_phases[j])
            diff = diff % (2 * math.pi)
            diff = min(diff, 2 * math.pi - diff)
            curvature[i, j] = diff
    
    curvature = curvature / (2 * math.pi)
    
    return curvature


def main():
    print("="*60)
    print("EXPERIMENT 3: CURVATURE PHASE TRANSITION")
    print("="*60)
    
    k = 7
    sigma_values = [0.0, 0.1, 0.2, 0.26, 0.28, 0.30, 0.32, 0.34, 0.4, 0.5, 0.6, 0.7]
    n_seeds = 10
    n_samples = 1000
    epochs = 150
    
    results = {}
    
    print(f"\nTraining k={k} at various noise levels...")
    print("-" * 60)
    
    for sigma in sigma_values:
        print(f"\nsigma = {sigma:.2f}")
        Q_values = []
        off_diag_values = []
        diag_values = []
        accuracies = []
        
        for seed in range(n_seeds):
            model = train_zk_with_noise(k, n_samples, epochs, seed=seed, noise_sigma=sigma)
            
            curvature = compute_curvature_matrix(model, k)
            Q, off_diag, diag = compute_Q(curvature)
            Q_values.append(Q)
            off_diag_values.append(off_diag)
            diag_values.append(diag)
            
            with torch.no_grad():
                x1, x2, y = generate_zk_data(k, 500)
                outputs = model(x1, x2)
                acc = (outputs.argmax(1) == y).float().mean().item()
                accuracies.append(acc)
        
        results[sigma] = {
            'Q_mean': np.mean(Q_values),
            'Q_std': np.std(Q_values),
            'off_diag_mean': np.mean(off_diag_values),
            'diag_mean': np.mean(diag_values),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'Q_values': Q_values,
            'accuracies': accuracies
        }
        
        print(f"  Q = {np.mean(Q_values):.4f} ± {np.std(Q_values):.4f}")
        print(f"  Off-diag = {np.mean(off_diag_values):.4f}, Diag = {np.mean(diag_values):.4f}")
        print(f"  Accuracy = {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
    
    print("\n" + "="*60)
    print("SUMMARY: Q vs sigma")
    print("="*60)
    print(f"\n{'sigma':>6} | {'Q':>8} | {'Q_std':>8} | {'Accuracy':>10}")
    print("-" * 40)
    for sigma in sigma_values:
        r = results[sigma]
        print(f"{sigma:6.2f} | {r['Q_mean']:8.4f} | {r['Q_std']:8.4f} | {r['accuracy_mean']:10.1%}")
    
    sigma_plot = list(results.keys())
    Q_means = [results[s]['Q_mean'] for s in sigma_plot]
    Q_stds = [results[s]['Q_std'] for s in sigma_plot]
    acc_means = [results[s]['accuracy_mean'] for s in sigma_plot]
    acc_stds = [results[s]['accuracy_std'] for s in sigma_plot]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.errorbar(sigma_plot, Q_means, yerr=Q_stds, fmt='o-', capsize=3, color='steelblue', linewidth=2, markersize=8)
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* ≈ 0.26')
    ax.set_xlabel('Training Noise (sigma)', fontsize=12)
    ax.set_ylabel('Q = ⟨|off-diag|⟩ - ⟨|diag|⟩', fontsize=12)
    ax.set_title('Curvature Order Parameter Q vs sigma', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.errorbar(sigma_plot, acc_means, yerr=acc_stds, fmt='s-', capsize=3, color='forestgreen', linewidth=2, markersize=8)
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* ≈ 0.26')
    ax.set_xlabel('Training Noise (sigma)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Training Noise', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    ax = axes[1, 0]
    off_diag_plot = [results[s]['off_diag_mean'] for s in sigma_plot]
    diag_plot = [results[s]['diag_mean'] for s in sigma_plot]
    ax.plot(sigma_plot, off_diag_plot, 'o-', label='⟨|off-diag|⟩', color='steelblue', linewidth=2, markersize=8)
    ax.plot(sigma_plot, diag_plot, 's--', label='⟨|diag|⟩', color='coral', linewidth=2, markersize=8)
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* ≈ 0.26')
    ax.set_xlabel('Training Noise (sigma)', fontsize=12)
    ax.set_ylabel('Curvature Value', fontsize=12)
    ax.set_title('Diagonal vs Off-Diagonal Curvature', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(sigma_plot, Q_means, 'o-', color='steelblue', linewidth=2, markersize=8, label='Q')
    l2 = ax2.plot(sigma_plot, acc_means, 's--', color='forestgreen', linewidth=2, markersize=8, label='Accuracy')
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Training Noise (sigma)', fontsize=12)
    ax.set_ylabel('Q', fontsize=12, color='steelblue')
    ax2.set_ylabel('Accuracy', fontsize=12, color='forestgreen')
    ax.set_title('Q and Accuracy Overlay', fontsize=14, fontweight='bold')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_curvature_phase_transition.png', dpi=300, bbox_inches='tight')
    print("\nSaved to experiment_curvature_phase_transition.png")
    
    sigma_critical = 0.26
    idx_below = sigma_plot.index(0.26) if 0.26 in sigma_plot else None
    if idx_below:
        Q_below = results[0.26]['Q_mean']
        Q_above = results[0.28]['Q_mean']
        drop = Q_below - Q_above
        print(f"\n*** PHASE TRANSITION ANALYSIS ***")
        print(f"Q at sigma=0.26: {Q_below:.4f}")
        print(f"Q at sigma=0.28: {Q_above:.4f}")
        print(f"Drop: {drop:.4f} ({(drop/Q_below)*100:.1f}%)")
        
        if drop > 0.1:
            print("\n*** SHARP TRANSITION CONFIRMED ***")
        else:
            print("\n*** GRADUAL TRANSITION ***")
    
    save_data = {str(k): v for k, v in results.items()}
    with open('experiment_curvature_phase_transition.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\nSaved to experiment_curvature_phase_transition.json")


if __name__ == "__main__":
    main()
