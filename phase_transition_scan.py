"""
Phase Transition Scan
=====================
Find exact n* where parity coherence is lost.
Tests: n = [32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96, 128]
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def get_lambda(epoch: int) -> float:
    if epoch < 20:
        return 0.0
    if epoch < 50:
        return 0.1
    if epoch < 100:
        return 0.3
    return 0.1


def generate_parity_dataset(n_bits: int, n_samples: int = 1000):
    if n_bits <= 12:
        n_possible = 2 ** n_bits
        X = torch.zeros(n_possible, n_bits)
        for i in range(n_possible):
            for j in range(n_bits):
                X[i, j] = (i >> j) & 1
        y = X.sum(dim=1) % 2
    else:
        X = torch.randint(0, 2, (n_samples, n_bits)).float()
        y = X.sum(dim=1) % 2
    return X.float(), y.float()


class HolonomyChainBundle(nn.Module):
    def __init__(self, n_bits: int):
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


def compute_kappa(phases):
    """Compute kappa (geometric order parameter)."""
    exp_vals = torch.exp(1j * phases)
    magnitude = torch.abs(exp_vals.mean())
    return 1 - magnitude


def train_bundle(model, X, y, epochs=300, lr=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        bce = nn.functional.binary_cross_entropy(outputs, y)
        hol_loss = model.compute_holonomy_loss()
        lam = get_lambda(epoch) * 2.0
        loss = bce + lam * hol_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}/{epochs}")
    
    return model


def run_single_n(n_bits, epochs=300):
    print(f"\n{'='*50}")
    print(f"Running n={n_bits}")
    print(f"{'='*50}")
    
    X, y = generate_parity_dataset(n_bits)
    print(f"Dataset: {len(X)} samples")
    
    model = HolonomyChainBundle(n_bits)
    model = train_bundle(model, X, y, epochs=epochs, lr=0.1)
    
    with torch.no_grad():
        outputs = model(X)
        accuracy = ((outputs > 0.5) == y).float().mean().item()
        kappa = compute_kappa(model.bit_phases)
        phase_mean = model.bit_phases.mean().item()
        phase_std = model.bit_phases.std().item()
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Phase mean: {phase_mean:.4f} (target: {math.pi:.4f})")
    print(f"Phase std: {phase_std:.4f}")
    print(f"Kappa: {kappa:.6f}")
    
    return {
        'n': n_bits,
        'accuracy': accuracy,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
        'kappa': kappa,
        'converged': phase_std < 0.1
    }


def main():
    n_values = [32, 40, 48, 52, 56, 57, 58, 59, 60, 64, 72, 80, 96, 128]
    
    results = []
    for n in n_values:
        r = run_single_n(n, epochs=300)
        results.append(r)
    
    print("\n" + "="*60)
    print("PHASE TRANSITION SCAN RESULTS")
    print("="*60)
    
    print("\n| n   | Accuracy | Phase Mean | Phase Std  | Kappa    | Converged |")
    print("|-----|----------|------------|------------|----------|-----------|")
    for r in results:
        conv_str = "YES" if r['converged'] else "NO"
        print(f"| {r['n']:3d} | {r['accuracy']:8.4f} | {r['phase_mean']:10.4f} | {r['phase_std']:10.4f} | {r['kappa']:8.6f} | {conv_str:9s} |")
    
    # Save JSON (convert numpy types to Python)
    results_json = []
    for r in results:
        results_json.append({
            'n': int(r['n']),
            'accuracy': float(r['accuracy']),
            'phase_mean': float(r['phase_mean']),
            'phase_std': float(r['phase_std']),
            'kappa': float(r['kappa']),
            'converged': bool(r['converged'])
        })
    with open('phase_transition_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print("\nSaved results to phase_transition_results.json")
    
    # Find n* (transition point)
    n_array = np.array([r['n'] for r in results])
    std_array = np.array([r['phase_std'] for r in results])
    acc_array = np.array([r['accuracy'] for r in results])
    
    # Find where phase_std jumps above 0.1
    transition_indices = np.where(std_array > 0.1)[0]
    if len(transition_indices) > 0:
        n_star_approx = n_array[transition_indices[0]]
        print(f"\nApproximate transition point n* ~ {n_star_approx}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy vs n
    ax1 = axes[0, 0]
    ax1.plot(n_array, acc_array, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlabel('n (number of bits)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Parity Accuracy vs n', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Phase std vs n (KEY PLOT)
    ax2 = axes[0, 1]
    ax2.plot(n_array, std_array, 'purple', linewidth=2, marker='o', markersize=8)
    ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Convergence threshold')
    ax2.set_xlabel('n (number of bits)', fontsize=12)
    ax2.set_ylabel('Phase Std (radians)', fontsize=12)
    ax2.set_title('Phase Std vs n (Order Parameter)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Log-log plot near transition for critical exponent
    ax3 = axes[1, 0]
    # Find approximate n* from the data
    n_star_est = n_array[np.argmax(std_array > 0.1)] if any(std_array > 0.1) else 64
    
    # Only plot points where n < n_star (to see approach to transition)
    mask = (n_array < n_star_est) & (n_star_est - n_array > 2)
    if np.sum(mask) >= 3:
        n_approach = n_star_est - n_array[mask]
        std_approach = std_array[mask]
        ax3.loglog(n_approach, std_approach, 'go-', linewidth=2, markersize=8)
        
        # Fit power law: std ~ (n* - n)^beta
        log_n = np.log(n_approach)
        log_std = np.log(std_approach + 1e-10)
        coeffs = np.polyfit(log_n, log_std, 1)
        beta = coeffs[0]
        ax3.set_xlabel('log(n* - n)', fontsize=12)
        ax3.set_ylabel('log(Phase Std)', fontsize=12)
        ax3.set_title(f'Critical Exponent Fit: β ≈ {beta:.2f}', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add fitted line
        x_fit = np.linspace(min(log_n), max(log_n), 100)
        y_fit = coeffs[0] * x_fit + coeffs[1]
        ax3.plot(np.exp(x_fit), np.exp(y_fit), 'r--', alpha=0.7, label=f'Fit: β={beta:.2f}')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Not enough points for critical exponent fit',
                ha='center', va='center', fontsize=14)
        ax3.set_xlabel('log(n* - n)')
        ax3.set_ylabel('log(Phase Std)')
    
    # Kappa vs n
    ax4 = axes[1, 1]
    kappa_array = np.array([r['kappa'] for r in results])
    ax4.plot(n_array, kappa_array, 'orange', linewidth=2, marker='s', markersize=8)
    ax4.set_xlabel('n (number of bits)', fontsize=12)
    ax4.set_ylabel('Kappa', fontsize=12)
    ax4.set_title('Kappa (Disorder) vs n', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_transition_scan.png', dpi=150, bbox_inches='tight')
    print("Saved plot to phase_transition_scan.png")
    plt.close()
    
    # Additional plot: combined view
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(n_array, std_array, 'purple', linewidth=2, marker='o', markersize=10, label='Phase Std')
    ax2_twin = ax.twinx()
    ax2_twin.plot(n_array, acc_array, 'blue', linewidth=2, marker='s', markersize=8, label='Accuracy')
    ax.axhline(y=0.1, color='purple', linestyle='--', alpha=0.3)
    ax.set_xlabel('n (number of bits)', fontsize=14)
    ax.set_ylabel('Phase Std (radians)', color='purple', fontsize=14)
    ax2_twin.set_ylabel('Accuracy', color='blue', fontsize=14)
    ax.set_title('Phase Transition: Order Parameter vs Performance', fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.savefig('phase_transition_combined.png', dpi=150, bbox_inches='tight')
    print("Saved combined plot to phase_transition_combined.png")
    plt.close()
    
    return results


if __name__ == "__main__":
    results = main()
