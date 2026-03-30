"""
Experiment 3: Curvature Phase Transition (v2)
==============================================
Instead of training with noise, train clean then measure curvature
stability when noise is applied at test time.
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


def train_zk(k, n_samples=1000, epochs=150, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1, x2, y = generate_zk_data(k, n_samples)
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model


def compute_curvature_variance(model, k, n_perturbations=50, noise_level=0.1):
    """Compute variance of curvature under perturbation"""
    output_phases = model.get_output_phases()
    
    base_curvature = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            diff = abs(output_phases[i] - output_phases[j])
            diff = diff % (2 * math.pi)
            diff = min(diff, 2 * math.pi - diff)
            base_curvature[i, j] = diff
    base_curvature = base_curvature / (2 * math.pi)
    
    curvatures = [base_curvature]
    
    for _ in range(n_perturbations):
        perturbed_phases = output_phases + np.random.randn(k) * noise_level
        perturbed_curvature = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                diff = abs(perturbed_phases[i] - perturbed_phases[j])
                diff = diff % (2 * math.pi)
                diff = min(diff, 2 * math.pi - diff)
                perturbed_curvature[i, j] = diff
        perturbed_curvature = perturbed_curvature / (2 * math.pi)
        curvatures.append(perturbed_curvature)
    
    curvatures = np.array(curvatures)
    variance = np.var(curvatures, axis=0).mean()
    
    return variance, base_curvature


def compute_Q_from_phases(output_phases):
    """Q = mean(|off-diagonal|) - mean(|diagonal|) from phase differences"""
    k = len(output_phases)
    phase_diffs = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            diff = abs(output_phases[i] - output_phases[j])
            diff = diff % (2 * math.pi)
            diff = min(diff, 2 * math.pi - diff)
            phase_diffs[i, j] = diff
    
    phase_diffs = phase_diffs / (2 * math.pi)
    
    diagonal = np.array([phase_diffs[i, i] for i in range(k)])
    off_diagonal = phase_diffs[~np.eye(k, dtype=bool)]
    
    Q = np.mean(np.abs(off_diagonal)) - np.mean(np.abs(diagonal))
    return Q


def main():
    print("="*60)
    print("EXPERIMENT 3: CURVATURE PHASE TRANSITION (v2)")
    print("="*60)
    
    k = 7
    n_seeds = 20
    n_samples = 1000
    epochs = 200
    
    sigma_test_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.26, 0.27, 0.28, 0.30, 0.35, 0.40, 0.50]
    
    results = {}
    
    print(f"\nTraining {n_seeds} models for k={k}...")
    
    models = []
    for seed in range(n_seeds):
        model = train_zk(k, n_samples, epochs, seed=seed)
        models.append(model)
    
    print(f"Measuring accuracy and curvature at various test noise levels...")
    print("-" * 60)
    
    for sigma_test in sigma_test_values:
        Q_values = []
        accuracies = []
        
        for model in models:
            output_phases = model.get_output_phases()
            Q = compute_Q_from_phases(output_phases)
            Q_values.append(Q)
            
            with torch.no_grad():
                x1, x2, y = generate_zk_data(k, 500)
                if sigma_test > 0:
                    outputs = model.forward_with_noise(x1, x2, sigma_test)
                else:
                    outputs = model(x1, x2)
                acc = (outputs.argmax(1) == y).float().mean().item()
                accuracies.append(acc)
        
        results[sigma_test] = {
            'Q_mean': np.mean(Q_values),
            'Q_std': np.std(Q_values),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'Q_values': Q_values,
            'accuracies': accuracies
        }
        
        print(f"sigma_test = {sigma_test:.2f}: Q = {np.mean(Q_values):.4f}, Acc = {np.mean(accuracies):.2%}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    sigma_plot = list(results.keys())
    Q_means = [results[s]['Q_mean'] for s in sigma_plot]
    Q_stds = [results[s]['Q_std'] for s in sigma_plot]
    acc_means = [results[s]['accuracy_mean'] for s in sigma_plot]
    acc_stds = [results[s]['accuracy_std'] for s in sigma_plot]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.errorbar(sigma_plot, Q_means, yerr=Q_stds, fmt='o-', capsize=3, color='steelblue', linewidth=2, markersize=8)
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* = 0.26')
    ax.set_xlabel('Test Noise (sigma)', fontsize=12)
    ax.set_ylabel('Q = <|off-diag|> - <|diag|>', fontsize=12)
    ax.set_title('Curvature Order Parameter Q vs Test Noise', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.errorbar(sigma_plot, acc_means, yerr=acc_stds, fmt='s-', capsize=3, color='forestgreen', linewidth=2, markersize=8)
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* = 0.26')
    ax.set_xlabel('Test Noise (sigma)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Test Noise', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(sigma_plot, Q_means, 'o-', color='steelblue', linewidth=2, markersize=8, label='Q')
    l2 = ax2.plot(sigma_plot, acc_means, 's--', color='forestgreen', linewidth=2, markersize=8, label='Accuracy')
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Test Noise (sigma)', fontsize=12)
    ax.set_ylabel('Q', fontsize=12, color='steelblue')
    ax2.set_ylabel('Accuracy', fontsize=12, color='forestgreen')
    ax.set_title('Q and Accuracy Overlay', fontsize=14, fontweight='bold')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for i, sigma in enumerate([0.0, 0.1, 0.2, 0.26, 0.3, 0.4]):
        if sigma in results:
            ax.hist(results[sigma]['accuracies'], bins=10, alpha=0.5, label=f'sigma={sigma}')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Accuracy Distribution Across Seeds', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_curvature_phase_transition_v2.png', dpi=300, bbox_inches='tight')
    print("\nSaved to experiment_curvature_phase_transition_v2.png")
    
    idx_26 = sigma_plot.index(0.26) if 0.26 in sigma_plot else None
    if idx_26:
        print(f"\n*** PHASE TRANSITION ANALYSIS ***")
        print(f"Accuracy at sigma=0.26: {results[0.26]['accuracy_mean']:.2%}")
        print(f"Q at sigma=0.26: {results[0.26]['Q_mean']:.4f}")
        
        if results[0.26]['accuracy_mean'] < 0.9:
            print("\n*** TRANSITION OCCURRING AT sigma* ***")
    
    save_data = {str(k): v for k, v in results.items()}
    with open('experiment_curvature_phase_transition_v2.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\nSaved to experiment_curvature_phase_transition_v2.json")


if __name__ == "__main__":
    main()
