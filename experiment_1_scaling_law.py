"""
Experiment 1: Complete the Scaling Law (k=3 fix)
===============================================
Previous scans used sigma in [0.0, 0.5], but k=3 has sigma* ~ 0.607 > 0.5
Re-run with sigma in [0.0, 0.85] to capture k=3.

Also run HIGH PRECISION at n=512, 50 seeds to identify the constant.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import json


class ZkBundle(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.input_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
        self.output_phases = nn.Parameter(torch.tensor([i * 2 * math.pi / k for i in range(k)]))
    
    def forward_no_noise(self, x1, x2):
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


def generate_zk_data(k, n_samples):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    return x1, x2, y


def train_zk(k, n_samples, epochs=150, seed=42):
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


def find_critical_sigma(model, x1, x2, y, sigma_range=(0.0, 0.85), n_points=30):
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], n_points)
    for sigma in sigma_values:
        with torch.no_grad():
            outputs = model.forward_with_noise(x1, x2, sigma)
            accuracy = (outputs.argmax(1) == y).float().mean().item()
        if accuracy < 0.8:
            return sigma, accuracy
    return None, 1.0


def main():
    print("="*60)
    print("EXPERIMENT 1: COMPLETE SCALING LAW (k=3 fix)")
    print("="*60)
    
    # Part 1: Extended range for k=3
    print("\n=== PART 1: Extended sigma range [0, 0.85] ===")
    
    k_values = [3, 5, 7, 11]
    n_values = [256, 512, 1024]
    n_seeds = 15
    
    results = {}
    
    for n in n_values:
        print(f"\n--- n = {n} ---")
        results[n] = {}
        
        for k in k_values:
            sigma_stars = []
            for seed in range(n_seeds):
                model, x1, x2, y = train_zk(k, n, seed=seed)
                sigma_candidate, _ = find_critical_sigma(model, x1, x2, y)
                sigma_stars.append(sigma_candidate)
            
            valid_sigmas = [s for s in sigma_stars if s is not None]
            mean_sigma = np.mean(valid_sigmas) if valid_sigmas else None
            results[n][k] = {
                'sigma_stars': sigma_stars,
                'mean_sigma_star': mean_sigma,
                'sigma_k': mean_sigma * k if mean_sigma else None
            }
            
            if mean_sigma:
                print(f"  k={k}: sigma* = {mean_sigma:.4f}, sigma* x k = {mean_sigma * k:.4f}")
            else:
                print(f"  k={k}: sigma* = N/A")
    
    print("\n" + "="*60)
    print("PART 1 SUMMARY: sigma* vs n")
    print("="*60)
    
    all_ratios = []
    for n in n_values:
        print(f"\nn = {n}:")
        for k in k_values:
            s = results[n][k]['mean_sigma_star']
            sk = results[n][k]['sigma_k']
            if s:
                print(f"  k={k}: sigma* = {s:.4f}, sigma* x k = {sk:.4f}")
                all_ratios.append(sk)
            else:
                print(f"  k={k}: sigma* = N/A")
    
    print(f"\nOverall sigma* x k: {np.mean(all_ratios):.4f} +/- {np.std(all_ratios):.4f}")
    
    in_range = [r for r in all_ratios if 1.70 <= r <= 1.95]
    if len(in_range) == len(all_ratios):
        print("\n*** SCALING LAW COMPLETE ***")
        print("All sigma* x k values within [1.70, 1.95]")
    else:
        print(f"\n*** SCALING LAW: {len(in_range)}/{len(all_ratios)} in range ***")
    
    # Part 2: HIGH PRECISION
    print("\n" + "="*60)
    print("PART 2: HIGH PRECISION (n=512, 50 seeds)")
    print("="*60)
    
    k_values_hp = [3, 5, 7, 11]
    n_seeds_hp = 50
    n_samples_hp = 512
    
    hp_results = {}
    
    for k in k_values_hp:
        print(f"\n  k = {k}...")
        sigma_stars = []
        
        for seed in range(n_seeds_hp):
            model, x1, x2, y = train_zk(k, n_samples_hp, seed=seed)
            sigma_candidate, _ = find_critical_sigma(
                model, x1, x2, y, 
                sigma_range=(0.0, 0.85), 
                n_points=50
            )
            sigma_stars.append(sigma_candidate)
        
        valid = [s for s in sigma_stars if s is not None]
        mean_sigma = np.mean(valid) if valid else None
        sigma_k = mean_sigma * k if mean_sigma else None
        
        hp_results[k] = {
            'sigma_stars': sigma_stars,
            'mean_sigma_star': mean_sigma,
            'sigma_k': sigma_k
        }
        
        print(f"    sigma* = {mean_sigma:.4f}, sigma* x k = {sigma_k:.4f}")
    
    # Calculate constant identity
    print("\n" + "="*60)
    print("CONSTANT IDENTITY ANALYSIS")
    print("="*60)
    
    sigma_k_values = [hp_results[k]['sigma_k'] for k in k_values_hp if hp_results[k]['sigma_k']]
    mean_constant = np.mean(sigma_k_values)
    std_constant = np.std(sigma_k_values)
    
    print(f"\nMeasured constant: {mean_constant:.4f} +/- {std_constant:.4f}")
    
    candidates = {
        'pi/sqrt(3)': math.pi / math.sqrt(3),
        '2*pi/sqrt(12)': 2 * math.pi / math.sqrt(12),
    }
    
    try:
        candidates['e^(pi/e)/pi'] = math.exp(math.pi / math.e) / math.pi
    except:
        pass
    
    print("\nCandidate comparison:")
    best_candidate = None
    best_residual = float('inf')
    
    for name, value in candidates.items():
        residual = abs(value - mean_constant)
        n_stds = residual / std_constant if std_constant > 0 else float('inf')
        print(f"  {name}: {value:.4f}, residual = {residual:.4f}, n_stds = {n_stds:.2f}")
        
        if residual < best_residual:
            best_residual = residual
            best_candidate = name
    
    print(f"\nClosest candidate: {best_candidate} = {candidates[best_candidate]:.4f}")
    print(f"Residual: {best_residual:.4f} ({best_residual/std_constant:.2f} std devs)")
    
    # Save results
    save_data = {
        'part1': results,
        'part2': {str(k): v for k, v in hp_results.items()},
        'constant_analysis': {
            'mean': mean_constant,
            'std': std_constant,
            'best_candidate': best_candidate,
            'candidate_value': candidates[best_candidate] if best_candidate else None,
            'residual': best_residual
        }
    }
    
    with open('experiment_1_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved to experiment_1_results.json")
    
    return results, hp_results


if __name__ == "__main__":
    main()
