"""
Experiment 4: Holonomy vs Accuracy Divergence
==============================================
k=11, sigma sweep from 0.0 to 0.35
Measure BOTH holonomy closure AND accuracy.
Find where they diverge.
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
    
    def get_input_phases(self):
        return self.input_phases.detach().cpu().numpy()
    
    def get_output_phases(self):
        return self.output_phases.detach().cpu().numpy()


def generate_zk_data(k, n_samples):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    return x1, x2, y


def train_zk(k, n_samples, epochs, seed):
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
    
    return model, x1, x2, y


def measure_holonomy_closure(model, k, n_tests=500):
    """Measure holonomy closure: does (x+y)-z = x+(y-z)?"""
    input_phases = model.get_input_phases()
    output_phases = model.get_output_phases()
    
    closures = 0
    total = 0
    
    for _ in range(n_tests):
        x = np.random.randint(0, k)
        y = np.random.randint(0, k)
        z = np.random.randint(0, k)
        
        phi_x = input_phases[x]
        phi_y = input_phases[y]
        phi_z = input_phases[z]
        
        phi_xy = (phi_x + phi_y) % (2 * math.pi)
        phi_yz = (phi_y + phi_z) % (2 * math.pi)
        
        phi_x_yz = (phi_x + phi_yz) % (2 * math.pi)
        phi_xy_z = (phi_xy + phi_z) % (2 * math.pi)
        
        dist = abs(phi_x_yz - phi_xy_z)
        dist = min(dist, 2*math.pi - dist)
        
        if dist < 0.1:
            closures += 1
        total += 1
    
    return closures / total


def measure_group_closure(model, k, n_tests=500):
    """Test if output correctly implements Z_k group operation"""
    correct = 0
    total = 0
    
    for _ in range(n_tests):
        x = np.random.randint(0, k)
        y = np.random.randint(0, k)
        
        with torch.no_grad():
            out = model.forward(torch.tensor([x]), torch.tensor([y]))
            pred = out.argmax().item()
            true = (x + y) % k
        
        if pred == true:
            correct += 1
        total += 1
    
    return correct / total


def main():
    print("="*60)
    print("EXPERIMENT 4: HOLONOMY vs ACCURACY DIVERGENCE")
    print("="*60)
    
    k = 11
    sigma_values = [0.0, 0.05, 0.10, 0.15, 0.17, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35]
    n_seeds = 20
    n_samples = 1000
    epochs = 200
    
    results = {}
    
    print(f"\nTraining {n_seeds} models for k={k}...")
    
    models = []
    for seed in range(n_seeds):
        model, x1, x2, y = train_zk(k, n_samples, epochs, seed=seed)
        models.append(model)
    
    print(f"Measuring holonomy closure and accuracy at various test noise levels...")
    print("-" * 60)
    
    for sigma in sigma_values:
        holonomy_closures = []
        group_closures = []
        accuracies = []
        
        for model in models:
            holonomy = measure_holonomy_closure(model, k, n_tests=200)
            group = measure_group_closure(model, k, n_tests=500)
            holonomy_closures.append(holonomy)
            group_closures.append(group)
            
            with torch.no_grad():
                x1, x2, y = generate_zk_data(k, 500)
                if sigma > 0:
                    outputs = model.forward_with_noise(x1, x2, sigma)
                else:
                    outputs = model(x1, x2)
                acc = (outputs.argmax(1) == y).float().mean().item()
                accuracies.append(acc)
        
        results[sigma] = {
            'holonomy_mean': np.mean(holonomy_closures),
            'holonomy_std': np.std(holonomy_closures),
            'group_closure_mean': np.mean(group_closures),
            'group_closure_std': np.std(group_closures),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'holonomy_values': holonomy_closures,
            'accuracy_values': accuracies
        }
        
        print(f"sigma = {sigma:.2f}: holonomy = {np.mean(holonomy_closures):.3f}, group = {np.mean(group_closures):.3f}, acc = {np.mean(accuracies):.2%}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    sigma_plot = list(results.keys())
    holonomy_means = [results[s]['holonomy_mean'] for s in sigma_plot]
    holonomy_stds = [results[s]['holonomy_std'] for s in sigma_plot]
    group_means = [results[s]['group_closure_mean'] for s in sigma_plot]
    group_stds = [results[s]['group_closure_std'] for s in sigma_plot]
    acc_means = [results[s]['accuracy_mean'] for s in sigma_plot]
    acc_stds = [results[s]['accuracy_std'] for s in sigma_plot]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.errorbar(sigma_plot, holonomy_means, yerr=holonomy_stds, fmt='o-', capsize=3, color='steelblue', linewidth=2, markersize=8, label='Holonomy Closure')
    ax.errorbar(sigma_plot, group_means, yerr=group_stds, fmt='s--', capsize=3, color='purple', linewidth=2, markersize=8, label='Group Closure')
    ax.axvline(x=0.17, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* = 0.17')
    ax.set_xlabel('Test Noise (sigma)', fontsize=12)
    ax.set_ylabel('Closure Rate', fontsize=12)
    ax.set_title('Holonomy vs Group Closure', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    ax = axes[0, 1]
    ax.errorbar(sigma_plot, acc_means, yerr=acc_stds, fmt='s-', capsize=3, color='forestgreen', linewidth=2, markersize=8)
    ax.axvline(x=0.17, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* = 0.17')
    ax.set_xlabel('Test Noise (sigma)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Test Noise', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(sigma_plot, holonomy_means, 'o-', color='steelblue', linewidth=2, markersize=8, label='Holonomy')
    l2 = ax2.plot(sigma_plot, acc_means, 's--', color='forestgreen', linewidth=2, markersize=8, label='Accuracy')
    ax.axvline(x=0.17, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Test Noise (sigma)', fontsize=12)
    ax.set_ylabel('Holonomy Closure', fontsize=12, color='steelblue')
    ax2.set_ylabel('Accuracy', fontsize=12, color='forestgreen')
    ax.set_title('Holonomy vs Accuracy', fontsize=14, fontweight='bold')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for i, sigma in enumerate([0.0, 0.1, 0.17, 0.25, 0.35]):
        if sigma in results:
            accs = results[sigma]['accuracy_values']
            ax.hist(accs, bins=10, alpha=0.5, label=f'sigma={sigma}')
    ax.axvline(x=0.8, color='black', linestyle=':', linewidth=2, label='80% threshold')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Accuracy Distribution Across Seeds', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_4_holonomy_accuracy.png', dpi=300, bbox_inches='tight')
    print("\nSaved to experiment_4_holonomy_accuracy.png")
    
    print("\n" + "="*60)
    print("DIVERGENCE ANALYSIS")
    print("="*60)
    
    sigma_star = 0.17
    idx = sigma_plot.index(sigma_star) if sigma_star in sigma_plot else None
    
    if idx:
        print(f"At sigma* = {sigma_star}:")
        print(f"  Holonomy closure: {holonomy_means[idx]:.3f}")
        print(f"  Group closure: {group_means[idx]:.3f}")
        print(f"  Accuracy: {acc_means[idx]:.2%}")
        
        holonomy_80 = [i for i, h in enumerate(holonomy_means) if h < 0.8]
        acc_80 = [i for i, a in enumerate(acc_means) if a < 0.8]
        
        if holonomy_80 and acc_80:
            print(f"\nHolonomy drops below 80% at sigma ~ {sigma_plot[holonomy_80[0]]}")
            print(f"Accuracy drops below 80% at sigma ~ {sigma_plot[acc_80[0]]}")
        elif acc_80:
            print(f"\nAccuracy drops below 80% at sigma ~ {sigma_plot[acc_80[0]]}")
            print("Holonomy stays robust (group structure maintained)")
    
    divergence_metric = [h - a for h, a in zip(holonomy_means, acc_means)]
    print(f"\nDivergence (holonomy - accuracy):")
    for s, d in zip(sigma_plot, divergence_metric):
        print(f"  sigma={s:.2f}: {d:.3f}")
    
    save_data = {str(k): v for k, v in results.items()}
    with open('experiment_4_holonomy_accuracy.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\nSaved to experiment_4_holonomy_accuracy.json")


if __name__ == "__main__":
    main()
