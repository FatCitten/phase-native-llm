"""
Z_k Critical Noise Experiment - Gaussian Noise on Phase Angles
===============================================================
Add Gaussian noise to the PHASE ENCODING (not input bits)
phi = 2*pi*x/k + N(0, sigma^2)

Expected: sigma* proportional to 1/k
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import json

def get_lambda(epoch):
    if epoch < 20:
        return 0.0
    if epoch < 50:
        return 0.1
    if epoch < 100:
        return 0.3
    return 0.1


class ZkBundle(nn.Module):
    """Z_k bundle with phase encoding."""
    
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
    
    def get_phases(self):
        return self.input_phases.detach().cpu().numpy()


def generate_zk_data(k, n_samples=2000):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    return x1, x2, y


def train_and_evaluate_zk(k, sigma=0.0, n_samples=1000, epochs=100, seed=42):
    """Train with Gaussian noise added to phase encoding."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1, x2, y = generate_zk_data(k, n_samples)
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Encode with Gaussian noise on phases
        if sigma > 0:
            # True phase encoding
            phi1_true = x1.float() * (2 * math.pi / k)
            phi2_true = x2.float() * (2 * math.pi / k)
            
            # Add Gaussian noise to phases
            phi1_noisy = phi1_true + torch.randn_like(phi1_true) * sigma
            phi2_noisy = phi2_true + torch.randn_like(phi2_true) * sigma
            
            # Map noisy phases back to indices (nearest)
            x1_idx = torch.clamp((phi1_noisy / (2 * math.pi) * k).long() % k, 0, k-1)
            x2_idx = torch.clamp((phi2_noisy / (2 * math.pi) * k).long() % k, 0, k-1)
        else:
            x1_idx, x2_idx = x1, x2
        
        outputs = model(x1_idx, x2_idx)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Evaluate on clean data
    with torch.no_grad():
        outputs = model(x1, x2)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    
    return accuracy


def main():
    print("="*60)
    print("Z_k CRITICAL NOISE - GAUSSIAN NOISE ON PHASES")
    print("="*60)
    
    k_values = [3, 5, 7, 11]
    sigma_values = np.linspace(0.0, 0.5, 15)
    n_seeds = 5
    
    results = {}
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Testing k={k}")
        print(f"{'='*50}")
        
        k_results = []
        
        for sigma in sigma_values:
            accs = []
            for seed in range(n_seeds):
                acc = train_and_evaluate_zk(k, sigma=sigma, seed=seed)
                accs.append(acc)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            k_results.append({
                'sigma': sigma,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'all_accs': accs
            })
            
            status = "OK" if mean_acc > 0.8 else "FAIL"
            print(f"  sigma={sigma:.3f}: acc={mean_acc:.3f} +/- {std_acc:.3f} [{status}]")
        
        # Find critical sigma (where accuracy drops below 80%)
        critical_sigma = None
        for r in k_results:
            if r['mean_acc'] < 0.8:
                critical_sigma = r['sigma']
                break
        
        results[k] = {
            'critical_sigma': critical_sigma,
            'data': k_results,
            'pi_over_k': math.pi / k
        }
        
        print(f"\n==> Critical sigma for k={k}: {critical_sigma}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n| k   | sigma*   | pi/k    | sigma* * k |")
    print("|------|----------|---------|------------|")
    
    for k, r in results.items():
        sigma_star = r['critical_sigma']
        pi_k = r['pi_over_k']
        ratio = sigma_star * k if sigma_star else float('nan')
        print(f"| {k:2d}  | {sigma_star:8.4f} | {pi_k:7.4f} |   {ratio:6.4f}   |")
    
    # Check proportionality
    print("\n" + "-"*60)
    sigma_star_values = [r['critical_sigma'] for r in results.values() if r['critical_sigma']]
    k_vals = [k for k in results.keys() if results[k]['critical_sigma']]
    
    if len(sigma_star_values) > 2:
        ratios = [s * k for s, k in zip(sigma_star_values, k_vals)]
        print(f"sigma* * k values: {[f'{r:.4f}' for r in ratios]}")
        print(f"Mean: {np.mean(ratios):.4f}, Std: {np.std(ratios):.4f}")
        
        if np.std(ratios) / np.mean(ratios) < 0.3:
            print("\n==> sigma* IS PROPORTIONAL TO 1/k!")
            print("    This is a strong theoretical result!")
    
    # Save
    save_results = {}
    for k, r in results.items():
        save_results[str(k)] = {
            'critical_sigma': r['critical_sigma'],
            'pi_over_k': r['pi_over_k'],
            'ratio': r['critical_sigma'] * k if r['critical_sigma'] else None
        }
    
    with open('zk_phase_noise_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print("\nSaved to zk_phase_noise_results.json")


if __name__ == "__main__":
    main()
