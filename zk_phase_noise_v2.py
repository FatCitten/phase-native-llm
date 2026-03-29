"""
Z_k Critical Noise - Gaussian noise on LEARNED PHASES
====================================================
Add noise to the phase encoding during forward pass (continuous)
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


class ZkBundleNoisy(nn.Module):
    """Z_k bundle with noisy phase encoding during training."""
    
    def __init__(self, k, noise_sigma=0.0):
        super().__init__()
        self.k = k
        self.noise_sigma = noise_sigma
        self.input_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
        self.output_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
    
    def forward(self, x1, x2, training=True):
        # Get base phases
        p1_base = self.input_phases[x1]
        p2_base = self.input_phases[x2]
        
        # Add noise during training
        if training and self.noise_sigma > 0:
            p1 = p1_base + torch.randn_like(p1_base) * self.noise_sigma
            p2 = p2_base + torch.randn_like(p2_base) * self.noise_sigma
        else:
            p1 = p1_base
            p2 = p2_base
        
        phi = (p1 + p2) % (2 * math.pi)
        dists = torch.abs(phi.unsqueeze(-1) - self.output_phases.unsqueeze(0))
        dists = dists % (2 * math.pi)
        dists = torch.min(dists, 2 * math.pi - dists)
        return -dists
    
    def get_phases(self):
        return self.input_phases.detach().cpu().numpy()


def generate_zk_data(k, n_samples=1000):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    return x1, x2, y


def train_and_evaluate_zk(k, sigma=0.0, n_samples=1000, epochs=150, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1, x2, y = generate_zk_data(k, n_samples)
    
    model = ZkBundleNoisy(k, noise_sigma=sigma)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Training with noise
        outputs = model(x1, x2, training=True)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Evaluate on clean data
    with torch.no_grad():
        outputs = model(x1, x2, training=False)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    
    return accuracy


def main():
    print("="*60)
    print("Z_k CRITICAL NOISE - NOISE ON LEARNED PHASES")
    print("="*60)
    
    k_values = [3, 5, 7, 11]
    sigma_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    n_seeds = 5
    
    results = {}
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        
        k_results = []
        
        for sigma in sigma_values:
            accs = []
            for seed in range(n_seeds):
                acc = train_and_evaluate_zk(k, sigma=sigma, seed=seed)
                accs.append(acc)
            
            mean_acc = np.mean(accs)
            k_results.append({
                'sigma': sigma,
                'mean_acc': mean_acc,
                'all_accs': accs
            })
            
            status = "OK" if mean_acc > 0.8 else "FAIL"
            print(f"  sigma={sigma:.2f}: acc={mean_acc:.3f} [{status}]")
        
        # Find critical sigma
        critical = None
        for r in k_results:
            if r['mean_acc'] < 0.8:
                critical = r['sigma']
                break
        
        results[k] = {
            'critical_sigma': critical,
            'data': k_results,
            'pi_over_k': math.pi / k
        }
        
        print(f"  ==> critical sigma for k={k}: {critical}")
    
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
    sigmas = [r['critical_sigma'] for r in results.values() if r['critical_sigma']]
    ks = [k for k, r in results.items() if r['critical_sigma']]
    
    if len(sigmas) > 2:
        ratios = [s * k for s, k in zip(sigmas, ks)]
        print(f"\nsigma* * k: {[f'{r:.4f}' for r in ratios]}")
        print(f"Mean: {np.mean(ratios):.4f}, Std: {np.std(ratios):.4f}")
    
    # Save
    save_results = {}
    for k, r in results.items():
        save_results[str(k)] = {
            'critical_sigma': r['critical_sigma'],
            'pi_over_k': r['pi_over_k']
        }
    
    with open('zk_phase_noise_v2.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print("\nSaved to zk_phase_noise_v2.json")


if __name__ == "__main__":
    main()
