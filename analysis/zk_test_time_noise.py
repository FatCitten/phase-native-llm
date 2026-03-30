"""
Z_k Critical Noise - TEST TIME ONLY
====================================
Train on STILL beam (no noise), then wobble at test time.
Find σ* where accuracy falls off a cliff.
Check if σ* ∝ 1/k.

This tests: "After training on a still beam, how much wobble before they fall?"
"""

import math
import torch
import torch.nn as nn
import numpy as np
import json


class ZkBundle(nn.Module):
    """Z_k bundle for modular arithmetic (no noise during training)."""
    
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.input_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
        self.output_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
    
    def forward_no_noise(self, x1, x2):
        """Forward pass without any noise."""
        p1 = self.input_phases[x1]
        p2 = self.input_phases[x2]
        phi = (p1 + p2) % (2 * math.pi)
        dists = torch.abs(phi.unsqueeze(-1) - self.output_phases.unsqueeze(0))
        dists = dists % (2 * math.pi)
        dists = torch.min(dists, 2 * math.pi - dists)
        return -dists
    
    def forward_with_noise(self, x1, x2, noise_sigma):
        """Forward pass with test-time noise added to phase encoding."""
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


def generate_zk_data(k, n_samples=1000):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    return x1, x2, y


def train_zk(k, n_samples=1000, epochs=150, seed=42):
    """Train Z_k bundle normally (no noise)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1, x2, y = generate_zk_data(k, n_samples)
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model.forward_no_noise(x1, x2)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model, x1, x2, y


def evaluate_with_noise(model, x1, x2, y, noise_sigma):
    """Evaluate model with test-time noise."""
    with torch.no_grad():
        outputs = model.forward_with_noise(x1, x2, noise_sigma)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    return accuracy


def find_critical_sigma(model, x1, x2, y, sigma_range=(0.0, 1.0), n_points=50):
    """Find sigma* where accuracy drops below 80%."""
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], n_points)
    
    for sigma in sigma_values:
        acc = evaluate_with_noise(model, x1, x2, y, sigma)
        if acc < 0.8:
            return sigma, acc
    return None, 1.0


def main():
    print("="*60)
    print("Z_k TEST-TIME NOISE EXPERIMENT")
    print("Train on still beam -> freeze -> wobble at test time")
    print("="*60)
    
    k_values = [3, 5, 7, 11]
    n_seeds = 5
    
    results = {}
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        
        sigma_stars = []
        acc_curves = []
        
        for seed in range(n_seeds):
            model, x1, x2, y = train_zk(k, seed=seed)
            
            sigma_candidate, acc_at_sigma = find_critical_sigma(model, x1, x2, y)
            sigma_stars.append(sigma_candidate)
            
            curve = []
            for sigma in np.linspace(0, 0.5, 21):
                acc = evaluate_with_noise(model, x1, x2, y, sigma)
                curve.append((sigma, acc))
            acc_curves.append(curve)
            
            print(f"  seed {seed}: sigma* = {sigma_candidate:.4f}")
        
        valid_sigmas = [s for s in sigma_stars if s is not None]
        if valid_sigmas:
            mean_sigma = np.mean(valid_sigmas)
        else:
            mean_sigma = None
            
        results[k] = {
            'sigma_stars': sigma_stars,
            'mean_sigma_star': mean_sigma,
            'acc_curves': acc_curves
        }
        
        print(f"  ==> mean sigma* for k={k}: {mean_sigma:.4f}" if mean_sigma else "  ==> no critical sigma found")
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print("\n| k   | sigma*   | pi/k    | sigma* * k |")
    print("|------|----------|---------|------------|")
    
    for k, r in results.items():
        sigma_star = r['mean_sigma_star']
        pi_k = math.pi / k
        if sigma_star:
            ratio = sigma_star * k
        else:
            ratio = float('nan')
        print(f"| {k:2d}  | {sigma_star if sigma_star else 'N/A':8.4f} | {pi_k:7.4f} |   {ratio:6.4f}   |")
    
    sigmas = [r['mean_sigma_star'] for r in results.values() if r['mean_sigma_star']]
    ks = [k for k, r in results.items() if r['mean_sigma_star']]
    
    if len(sigmas) > 2:
        ratios = [s * k for s, k in zip(sigmas, ks)]
        print(f"\nsigma* * k: {[f'{r:.4f}' for r in ratios]}")
        print(f"Mean: {np.mean(ratios):.4f}, Std: {np.std(ratios):.4f}")
        
        if np.std(ratios) / np.mean(ratios) < 0.3:
            print("\n*** PROPORTIONALITY CONFIRMED: sigma* ~ 1/k ***")
        else:
            print("\n*** NO CLEAR PROPORTIONALITY ***")
    
    save_data = {}
    for k, r in results.items():
        save_data[str(k)] = {
            'mean_sigma_star': r['mean_sigma_star'],
            'all_sigma_stars': r['sigma_stars']
        }
    
    with open('zk_test_time_noise_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved to zk_test_time_noise_results.json")


if __name__ == "__main__":
    main()
