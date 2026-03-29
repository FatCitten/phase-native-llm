"""
Refined Phase Transition Scan - Multiple Seeds
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def get_lambda(epoch):
    if epoch < 20:
        return 0.0
    if epoch < 50:
        return 0.1
    if epoch < 100:
        return 0.3
    return 0.1


def generate_parity_dataset(n_bits, n_samples=2000):
    X = torch.randint(0, 2, (n_samples, n_bits)).float()
    y = X.sum(dim=1) % 2
    return X.float(), y.float()


class HolonomyChainBundle(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.bit_phases = nn.Parameter(
            torch.ones(n_bits) * math.pi + torch.randn(n_bits) * 0.1
        )
        self.phi_0 = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.ones(n_bits) * math.pi)
    
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        phi = self.phi_0.expand(batch_size)
        for i in range(self.n_bits):
            phi = phi + inputs[:, i] * self.bit_phases[i]
        output = (1.0 - torch.cos(phi)) / 2.0
        return output
    
    def compute_holonomy_loss(self):
        R_actual = torch.exp(1j * self.bit_phases)
        R_predicted = torch.exp(1j * self.A)
        return torch.mean(torch.abs(R_actual - R_predicted).pow(2))


def train_and_evaluate(n_bits, n_samples=2000, epochs=400, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X, y = generate_parity_dataset(n_bits, n_samples)
    
    model = HolonomyChainBundle(n_bits)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        bce = nn.functional.binary_cross_entropy(outputs, y)
        hol_loss = model.compute_holonomy_loss()
        lam = get_lambda(epoch) * 2.0
        loss = bce + lam * hol_loss
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        outputs = model(X)
        accuracy = ((outputs > 0.5) == y).float().mean().item()
        phase_std = model.bit_phases.std().item()
    
    return accuracy, phase_std


def main():
    n_values = [32, 40, 48, 52, 56, 58, 60, 64, 72, 80]
    n_seeds = 5
    n_samples = 2000
    epochs = 400
    
    results = []
    
    for n in n_values:
        accs = []
        stds = []
        for seed in range(n_seeds):
            acc, std = train_and_evaluate(n, n_samples, epochs, seed)
            accs.append(acc)
            stds.append(std)
            print(f"n={n}, seed={seed}: acc={acc:.4f}, std={std:.4f}")
        
        results.append({
            'n': n,
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
            'mean_phase_std': np.mean(stds),
            'all_accuracies': accs,
            'all_phase_stds': stds
        })
    
    print("\n" + "="*60)
    print("REFINED RESULTS (5 seeds each)")
    print("="*60)
    
    print("\n| n   | Mean Acc | Std Acc | Mean Phase Std |")
    print("|-----|----------|---------|----------------|")
    for r in results:
        print(f"| {r['n']:3d} | {r['mean_accuracy']:8.4f} | {r['std_accuracy']:6.4f} | {r['mean_phase_std']:14.4f} |")
    
    # Save results
    with open('refined_transition_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    n_array = np.array([r['n'] for r in results])
    acc_array = np.array([r['mean_accuracy'] for r in results])
    std_array = np.array([r['mean_phase_std'] for r in results])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.errorbar(n_array, acc_array, yerr=[r['std_accuracy'] for r in results], 
                 fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlabel('n (bits)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Parity Accuracy vs n (5 seeds each)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(n_array, std_array, 'purple', marker='o', linewidth=2, markersize=8)
    ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('n (bits)')
    ax2.set_ylabel('Phase Std')
    ax2.set_title('Phase Std vs n (Order Parameter)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('refined_transition.png', dpi=150)
    print("\nSaved plot to refined_transition.png")
    plt.close()
    
    return results


if __name__ == "__main__":
    results = main()
