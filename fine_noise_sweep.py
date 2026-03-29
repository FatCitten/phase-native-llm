"""
Fine-grained Noise Sweep on Synthetic Parity
============================================
Test σ = 0.0, 0.01, 0.02, 0.05, 0.1, 0.2
Question: Sharp or gradual transition from 100% to 50% accuracy?
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


def generate_parity_data(n_bits, n_samples=2000):
    """Generate clean binary parity data."""
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


def train_and_evaluate(n_bits, noise_sigma=0.0, n_samples=2000, epochs=400, seed=42):
    """Train with Gaussian noise added to inputs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X, y = generate_parity_data(n_bits, n_samples)
    
    model = HolonomyChain(n_bits)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        # Apply Gaussian noise to inputs during training
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
    
    # Evaluate on clean data
    with torch.no_grad():
        outputs = model(X)
        accuracy = ((outputs > 0.5) == y).float().mean().item()
        phases = model.get_phases()
        phase_mean = np.mean(phases)
        phase_std = np.std(phases)
    
    return accuracy, phase_mean, phase_std, phases


def main():
    print("="*60)
    print("FINE-GRAINED NOISE SWEEP ON SYNTHETIC PARITY")
    print("="*60)
    
    n_bits = 32
    noise_sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    n_seeds = 5
    
    results = []
    
    for sigma in noise_sigmas:
        accs = []
        means = []
        stds = []
        
        print(f"\n--- Sigma = {sigma} ---")
        for seed in range(n_seeds):
            acc, mean, std, _ = train_and_evaluate(n_bits, sigma, seed=seed)
            accs.append(acc)
            means.append(mean)
            stds.append(std)
            print(f"  Seed {seed}: acc={acc:.4f}, mean={mean:.4f}, std={std:.4f}")
        
        results.append({
            'sigma': sigma,
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
            'mean_phase_mean': np.mean(means),
            'mean_phase_std': np.mean(stds),
            'all_accuracies': accs,
            'all_phase_means': means,
            'all_phase_stds': stds
        })
    
    # Print results table
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70)
    print("\n| Sigma  | Accuracy | Mean Phase | Phase Std |")
    print("|--------|----------|------------|-----------|")
    for r in results:
        print(f"| {r['sigma']:6.3f} |   {r['mean_accuracy']:.4f}  |   {r['mean_phase_mean']:7.4f}  |  {r['mean_phase_std']:.4f}  |")
    
    # Determine transition type
    print("\n" + "-"*70)
    print("TRANSITION ANALYSIS:")
    print("-"*70)
    
    accs = [r['mean_accuracy'] for r in results]
    sigmas = [r['sigma'] for r in results]
    
    # Find where accuracy drops below 90%, 75%, 50%
    for threshold in [0.9, 0.75, 0.5]:
        for i, acc in enumerate(accs):
            if acc < threshold:
                print(f"Accuracy drops below {threshold*100:.0f}% at sigma ~ {sigmas[i]}")
                break
    
    # Check if transition is sharp or gradual
    acc_diff = []
    for i in range(1, len(accs)):
        diff = abs(accs[i] - accs[i-1])
        acc_diff.append((sigmas[i] - sigmas[i-1], diff))
    
    print(f"\nAccuracy changes between consecutive sigma values:")
    for ds, da in acc_diff:
        rate = da / ds if ds > 0 else 0
        print(f"  sigma change {ds:.3f}: accuracy change {da:.4f} (rate: {rate:.2f})")
    
    # Determine transition type
    max_jump = max([da for ds, da in acc_diff])
    if max_jump > 0.3:
        print(f"\n==> SHARP TRANSITION (max jump = {max_jump:.4f})")
        print("    This suggests a FIRST-ORDER phase transition")
    else:
        print(f"\n==> GRADUAL TRANSITION (max jump = {max_jump:.4f})")
        print("    This suggests a SECOND-ORDER phase transition")
    
    # Plot
    sigma_arr = np.array([r['sigma'] for r in results])
    acc_arr = np.array([r['mean_accuracy'] for r in results])
    acc_err = np.array([r['std_accuracy'] for r in results])
    mean_arr = np.array([r['mean_phase_mean'] for r in results])
    std_arr = np.array([r['mean_phase_std'] for r in results])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy vs sigma
    ax1 = axes[0, 0]
    ax1.errorbar(sigma_arr, acc_arr, yerr=acc_err, fmt='b-o', capsize=5, linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax1.axhline(y=0.9, color='g', linestyle=':', alpha=0.5, label='90% threshold')
    ax1.set_xlabel('Noise Sigma')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Parity Accuracy vs Input Noise (Fine Sweep)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Phase mean vs sigma
    ax2 = axes[0, 1]
    ax2.plot(sigma_arr, mean_arr, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=math.pi, color='r', linestyle='--', alpha=0.5, label=f'pi={math.pi:.2f}')
    ax2.set_xlabel('Noise Sigma')
    ax2.set_ylabel('Mean Phase (radians)')
    ax2.set_title('Mean Phase vs Input Noise')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Phase std vs sigma (KEY PLOT - order parameter)
    ax3 = axes[1, 0]
    ax3.plot(sigma_arr, std_arr, 'purple', marker='o', linewidth=2, markersize=8)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Noise Sigma')
    ax3.set_ylabel('Phase Std (radians)')
    ax3.set_title('Phase Std (Order Parameter) vs Input Noise')
    ax3.grid(True, alpha=0.3)
    
    # Combined view
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    l1 = ax4.plot(sigma_arr, std_arr, 'purple', marker='o', linewidth=2, markersize=8, label='Phase Std')
    l2 = ax4_twin.plot(sigma_arr, acc_arr, 'blue', marker='s', linewidth=2, markersize=8, label='Accuracy')
    ax4.set_xlabel('Noise Sigma')
    ax4.set_ylabel('Phase Std', color='purple')
    ax4_twin.set_ylabel('Accuracy', color='blue')
    ax4.set_title('Phase Disorder vs Performance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fine_noise_sweep.png', dpi=150)
    print("\nSaved plot to fine_noise_sweep.png")
    plt.close()
    
    # Save JSON
    results_json = []
    for r in results:
        results_json.append({
            'sigma': float(r['sigma']),
            'mean_accuracy': float(r['mean_accuracy']),
            'std_accuracy': float(r['std_accuracy']),
            'mean_phase_mean': float(r['mean_phase_mean']),
            'mean_phase_std': float(r['mean_phase_std'])
        })
    
    with open('fine_noise_sweep_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print("Saved results to fine_noise_sweep_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
