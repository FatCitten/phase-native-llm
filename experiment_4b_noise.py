"""
Experiment 4B: Noise Injection on Synthetic Parity (Fixed)
========================================================
Test how Gaussian noise affects phase learning.
Keep inputs binary, just add noise during training (like label noise or input dropout).
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


def train_and_evaluate(n_bits, noise_prob=0.0, n_samples=2000, epochs=400, seed=42):
    """Train with input dropout (probability of flipping each bit)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X, y = generate_parity_data(n_bits, n_samples)
    
    model = HolonomyChain(n_bits)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        # Apply input noise during training
        X_noisy = X.clone()
        if noise_prob > 0:
            flip_mask = torch.rand_like(X_noisy) < noise_prob
            X_noisy = torch.where(flip_mask, 1 - X_noisy, X_noisy)
        
        optimizer.zero_grad()
        outputs = model(X_noisy)
        bce = nn.functional.binary_cross_entropy(outputs, y)
        hol_loss = model.compute_holonomy_loss()
        lam = get_lambda(epoch) * 2.0
        loss = bce + lam * hol_loss
        loss.backward()
        optimizer.step()
    
    # Evaluate on noisy data (same as training)
    with torch.no_grad():
        X_test = X.clone()
        if noise_prob > 0:
            flip_mask = torch.rand_like(X_test) < noise_prob
            X_test = torch.where(flip_mask, 1 - X_test, X_test)
        outputs = model(X_test)
        accuracy = ((outputs > 0.5) == y).float().mean().item()
        phases = model.get_phases()
        phase_mean = np.mean(phases)
        phase_std = np.std(phases)
    
    return accuracy, phase_mean, phase_std, phases


def main():
    print("="*60)
    print("EXPERIMENT 4B: NOISE INJECTION ON SYNTHETIC PARITY")
    print("(Using input bit flip probability)")
    print("="*60)
    
    n_bits = 32
    noise_probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    n_seeds = 5
    
    results = []
    
    for noise_prob in noise_probs:
        accs = []
        means = []
        stds = []
        
        print(f"\n--- Flip Probability = {noise_prob} ---")
        for seed in range(n_seeds):
            acc, mean, std, _ = train_and_evaluate(n_bits, noise_prob, seed=seed)
            accs.append(acc)
            means.append(mean)
            stds.append(std)
            print(f"  Seed {seed}: acc={acc:.4f}, mean={mean:.4f}, std={std:.4f}")
        
        results.append({
            'noise_prob': noise_prob,
            'mean_accuracy': np.mean(accs),
            'mean_phase_mean': np.mean(means),
            'mean_phase_std': np.mean(stds),
            'std_accuracy': np.std(accs),
            'all_accuracies': accs,
            'all_phase_means': means,
            'all_phase_stds': stds
        })
    
    # Print results table
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70)
    print("\n| Flip Prob | Accuracy | Mean Phase | Phase Std |")
    print("|-----------|----------|------------|-----------|")
    for r in results:
        print(f"|    {r['noise_prob']:5.2f}  |   {r['mean_accuracy']:.4f}  |   {r['mean_phase_mean']:7.4f}  |  {r['mean_phase_std']:.4f}  |")
    
    # Compare with MNIST results
    print("\n" + "-"*70)
    print("COMPARISON WITH MNIST:")
    print("-"*70)
    mnist_n32 = {'mean_phase_mean': 3.60, 'mean_phase_std': 0.23, 'test_acc': 0.848}
    print(f"\nSynthetic clean (flip=0.0): mean={results[0]['mean_phase_mean']:.4f}, std={results[0]['mean_phase_std']:.4f}, acc={results[0]['mean_accuracy']:.4f}")
    print(f"MNIST (linear projection):  mean={mnist_n32['mean_phase_mean']:.4f}, std={mnist_n32['mean_phase_std']:.4f}, acc={mnist_n32['test_acc']:.4f}")
    
    # Plot
    prob_arr = np.array([r['noise_prob'] for r in results])
    acc_arr = np.array([r['mean_accuracy'] for r in results])
    mean_arr = np.array([r['mean_phase_mean'] for r in results])
    std_arr = np.array([r['mean_phase_std'] for r in results])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy vs noise
    ax1 = axes[0, 0]
    ax1.errorbar(prob_arr, acc_arr, yerr=[r['std_accuracy'] for r in results], fmt='b-o', capsize=5, linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlabel('Bit Flip Probability')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Parity Accuracy vs Input Noise')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Phase mean vs noise
    ax2 = axes[0, 1]
    ax2.plot(prob_arr, mean_arr, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=math.pi, color='r', linestyle='--', alpha=0.5, label=f'pi={math.pi:.2f}')
    ax2.axhline(y=mnist_n32['mean_phase_mean'], color='purple', linestyle=':', alpha=0.7, label=f'MNIST mean={mnist_n32["mean_phase_mean"]:.2f}')
    ax2.set_xlabel('Bit Flip Probability')
    ax2.set_ylabel('Mean Phase (radians)')
    ax2.set_title('Mean Phase vs Input Noise')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Phase std vs noise
    ax3 = axes[1, 0]
    ax3.plot(prob_arr, std_arr, 'purple', marker='o', linewidth=2, markersize=8)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=mnist_n32['mean_phase_std'], color='purple', linestyle=':', alpha=0.7, label=f'MNIST std={mnist_n32["mean_phase_std"]:.2f}')
    ax3.set_xlabel('Bit Flip Probability')
    ax3.set_ylabel('Phase Std (radians)')
    ax3.set_title('Phase Std (Order Parameter) vs Input Noise')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Combined view
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    l1 = ax4.plot(prob_arr, std_arr, 'purple', marker='o', linewidth=2, markersize=8, label='Phase Std')
    l2 = ax4_twin.plot(prob_arr, acc_arr, 'blue', marker='s', linewidth=2, markersize=8, label='Accuracy')
    ax4.set_xlabel('Bit Flip Probability')
    ax4.set_ylabel('Phase Std', color='purple')
    ax4_twin.set_ylabel('Accuracy', color='blue')
    ax4.set_title('Phase Disorder vs Performance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_injection_results.png', dpi=150)
    print("\nSaved plot to noise_injection_results.png")
    plt.close()
    
    # Save JSON
    results_json = []
    for r in results:
        results_json.append({
            'noise_prob': float(r['noise_prob']),
            'mean_accuracy': float(r['mean_accuracy']),
            'mean_phase_mean': float(r['mean_phase_mean']),
            'mean_phase_std': float(r['mean_phase_std']),
            'std_accuracy': float(r['std_accuracy'])
        })
    
    with open('noise_injection_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print("Saved results to noise_injection_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
