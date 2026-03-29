"""
Critical Point Analysis: Bimodal vs Unimodal Distribution
==========================================================
Run 20 seeds at sigma = 0.07 (critical point)
If first-order: bimodal distribution (peaks at 100% and 50%)
If second-order: unimodal distribution (peak around 75%)
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_lambda(epoch):
    if epoch < 20:
        return 0.0
    if epoch < 50:
        return 0.1
    if epoch < 100:
        return 0.3
    return 0.1


def generate_parity_data(n_bits, n_samples=2000):
    X = torch.randint(0, 2, (n_samples, n_bits)).float()
    y = X.sum(dim=1) % 2
    return X.float(), y.float()


class HolonomyChain(nn.Module):
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
    
    def get_phases(self):
        return self.bit_phases.detach().cpu().numpy()


def train_and_evaluate(n_bits, noise_sigma, n_samples=2000, epochs=400, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X, y = generate_parity_data(n_bits, n_samples)
    
    model = HolonomyChain(n_bits)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        X_noisy = X + torch.randn_like(X) * noise_sigma
        X_noisy = torch.clamp(X_noisy, 0, 1)
        
        optimizer.zero_grad()
        outputs = model(X_noisy)
        bce = nn.functional.binary_cross_entropy(outputs, y)
        hol_loss = model.compute_holonomy_loss()
        lam = get_lambda(epoch) * 2.0
        loss = bce + lam * hol_loss
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        outputs = model(X)
        accuracy = ((outputs > 0.5) == y).float().mean().item()
        phases = model.get_phases()
        phase_mean = np.mean(phases)
        phase_std = np.std(phases)
    
    return accuracy, phase_mean, phase_std


def main():
    print("="*60)
    print("CRITICAL POINT ANALYSIS: sigma = 0.07")
    print("Testing 20 seeds for bimodality")
    print("="*60)
    
    n_bits = 32
    sigma = 0.07
    n_seeds = 20
    epochs = 400
    
    accuracies = []
    phase_means = []
    phase_stds = []
    
    print(f"\nRunning {n_seeds} seeds at sigma = {sigma}...")
    for seed in range(n_seeds):
        acc, mean, std = train_and_evaluate(n_bits, sigma, seed=seed, epochs=epochs)
        accuracies.append(acc)
        phase_means.append(mean)
        phase_stds.append(std)
        
        status = "SUCCESS" if acc > 0.9 else "FAILED" if acc < 0.6 else "MIXED"
        print(f"  Seed {seed:2d}: accuracy = {acc:.4f}, mean = {mean:.4f}, std = {std:.4f} [{status}]")
    
    # Statistics
    acc_array = np.array(accuracies)
    mean_array = np.array(phase_means)
    std_array = np.array(phase_stds)
    
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Mean accuracy: {np.mean(acc_array):.4f}")
    print(f"Std accuracy:  {np.std(acc_array):.4f}")
    print(f"Min accuracy:  {np.min(acc_array):.4f}")
    print(f"Max accuracy:  {np.max(acc_array):.4f}")
    
    n_success = np.sum(acc_array > 0.9)
    n_failed = np.sum(acc_array < 0.6)
    n_mixed = len(acc_array) - n_success - n_failed
    
    print(f"\nSuccess (>90%): {n_success}/20")
    print(f"Mixed (60-90%): {n_mixed}/20")
    print(f"Failed (<60%): {n_failed}/20")
    
    # Determine bimodality
    print("\n" + "-"*60)
    print("DISTRIBUTION ANALYSIS:")
    print("-"*60)
    
    if n_success >= 5 and n_failed >= 5 and n_mixed <= 3:
        print("==> BIMODAL DISTRIBUTION CONFIRMED!")
        print("    This is the SIGNATURE of a FIRST-ORDER PHASE TRANSITION")
        print("    Multiple stable states coexist at the critical point")
    elif np.std(acc_array) > 0.2:
        print("==> HIGH VARIANCE - possibly bimodal")
        print("    More seeds needed for definitive conclusion")
    else:
        print("==> UNIMODAL DISTRIBUTION")
        print("    This would suggest a SECOND-ORDER PHASE TRANSITION")
    
    # Plot histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram of accuracies
    ax1 = axes[0]
    ax1.hist(acc_array, bins=10, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0.5, color='r', linestyle='--', label='Random (50%)')
    ax1.axvline(x=1.0, color='g', linestyle='--', label='Perfect (100%)')
    ax1.axvline(x=np.mean(acc_array), color='blue', linestyle='-', label=f'Mean ({np.mean(acc_array):.2f})')
    ax1.set_xlabel('Accuracy')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Accuracy Distribution at Critical Point (sigma={sigma})')
    ax1.legend()
    ax1.set_xlim(0, 1.05)
    
    # Histogram of phase means
    ax2 = axes[1]
    ax2.hist(mean_array, bins=10, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(x=math.pi, color='r', linestyle='--', label=f'pi ({math.pi:.2f})')
    ax2.set_xlabel('Mean Phase (radians)')
    ax2.set_ylabel('Count')
    ax2.set_title('Phase Mean Distribution')
    ax2.legend()
    
    # Histogram of phase stds
    ax3 = axes[2]
    ax3.hist(std_array, bins=10, edgecolor='black', alpha=0.7, color='purple')
    ax3.axvline(x=0, color='r', linestyle='--', label='Ordered (0)')
    ax3.axvline(x=1.2, color='orange', linestyle='--', label='Disordered (~1.2)')
    ax3.set_xlabel('Phase Std')
    ax3.set_ylabel('Count')
    ax3.set_title('Phase Std Distribution')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('critical_point_bimodality.png', dpi=150)
    print("\nSaved plot to critical_point_bimodality.png")
    plt.close()
    
    # Scatter plot: accuracy vs phase std
    fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(std_array, acc_array, s=100, alpha=0.7, c='blue', edgecolors='black')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Phase Std (Order Parameter)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy vs Phase Std at Critical Point (sigma={sigma})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('critical_point_scatter.png', dpi=150)
    print("Saved plot to critical_point_scatter.png")
    plt.close()
    
    return accuracies, phase_means, phase_stds


if __name__ == "__main__":
    accuracies, phase_means, phase_stds = main()
