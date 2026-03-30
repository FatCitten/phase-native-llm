"""
NEW-D: Identify the Constant (H1 vs H2) - FIXED
================================================
Test whether sigma* x k = constant (~1.944) or follows 1.944 - 0.623/k.

FIXED: Use finer sigma grid to avoid std=0 quantization artifact
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


def find_critical_sigma_continuous(model, x1, x2, y, sigma_range=(0.0, 1.0), n_points=200):
    """Use finer grid to avoid quantization"""
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], n_points)
    accuracies = []
    
    for sigma in sigma_values:
        with torch.no_grad():
            outputs = model.forward_with_noise(x1, x2, sigma)
            accuracy = (outputs.argmax(1) == y).float().mean().item()
        accuracies.append(accuracy)
    
    accuracies = np.array(accuracies)
    
    below_80 = np.where(accuracies < 0.8)[0]
    if len(below_80) > 0:
        first_idx = below_80[0]
        if first_idx > 0:
            return float(sigma_values[first_idx]), float(accuracies[first_idx])
        return float(sigma_values[0]), float(accuracies[0])
    
    return None, 1.0


def main():
    print("="*60)
    print("NEW-D: IDENTIFY THE CONSTANT (FIXED)")
    print("="*60)
    print("\nH1: sigma* x k = constant (~1.944)")
    print("H2: sigma* x k = 1.944 - 0.623/k")
    print("FIX: Finer sigma grid (200 points) to avoid std=0")
    print("="*60)
    
    k_values = [3, 5, 7, 11, 13, 17, 19, 23, 29]
    n_seeds = 50
    n_samples = 1000
    epochs = 150
    
    results = {}
    
    for k in k_values:
        print(f"\n--- k = {k} ({n_seeds} seeds) ---")
        sigma_stars = []
        
        for seed in range(n_seeds):
            model, x1, x2, y = train_zk(k, n_samples, epochs=epochs, seed=seed)
            sigma_candidate, _ = find_critical_sigma_continuous(
                model, x1, x2, y,
                sigma_range=(0.0, 1.0),
                n_points=200
            )
            sigma_stars.append(sigma_candidate)
            
            if (seed + 1) % 25 == 0:
                valid = [s for s in sigma_stars if s is not None]
                if valid:
                    current_mean = np.mean(valid)
                    print(f"  {seed+1}/{n_seeds}: current mean sigma* = {current_mean:.4f}, C(k) = {current_mean*k:.4f}")
        
        valid_sigmas = [s for s in sigma_stars if s is not None]
        mean_sigma = np.mean(valid_sigmas) if valid_sigmas else None
        sigma_k = mean_sigma * k if mean_sigma else None
        sigma_k_std = np.std([s * k for s in valid_sigmas]) if len(valid_sigmas) > 1 else 0.0
        
        results[k] = {
            'sigma_stars': sigma_stars,
            'mean_sigma_star': mean_sigma,
            'sigma_k': sigma_k,
            'sigma_k_std': sigma_k_std,
            'n_valid': len(valid_sigmas)
        }
        
        print(f"  Final: sigma* = {mean_sigma:.4f} +/- {np.std(valid_sigmas):.4f}, C(k) = {sigma_k:.4f} +/- {sigma_k_std:.4f}")
    
    print("\n" + "="*60)
    print("RAW C(k) DATA")
    print("="*60)
    print(f"{'k':>4} | {'C(k)':>8} | {'std':>8}")
    print("-" * 30)
    
    for k in k_values:
        r = results[k]
        print(f"{k:>4} | {r['sigma_k']:>8.4f} | {r['sigma_k_std']:>8.4f}")
    
    print("\n" + "="*60)
    print("HYPOTHESIS TEST")
    print("="*60)
    
    sigma_k_values = [results[k]['sigma_k'] for k in k_values]
    sigma_k_stds = [max(results[k]['sigma_k_std'], 0.001) for k in k_values]  # Floor to avoid div by zero
    
    h1_constant = 1.944
    h1_residuals = [abs(sigma_k_values[i] - h1_constant) / sigma_k_stds[i] 
                   for i in range(len(k_values))]
    h1_avg_residual = np.mean(h1_residuals)
    
    print(f"\nH1 (constant = 1.944): avg residual = {h1_avg_residual:.2f} std devs")
    
    h2_predictions = [1.944 - 0.623/k for k in k_values]
    h2_residuals = [(sigma_k_values[i] - h2_predictions[i]) / sigma_k_stds[i] 
                    for i in range(len(k_values))]
    h2_avg_residual = np.mean(h2_residuals)
    
    print(f"H2 (1.944 - 0.623/k): avg residual = {h2_avg_residual:.2f} std devs")
    
    if h2_avg_residual < h1_avg_residual:
        print("\n*** RESULT: H2 WINS (1.944 - 0.623/k) ***")
    elif h1_avg_residual < h2_avg_residual:
        print("\n*** RESULT: H1 WINS (constant ~1.944) ***")
    else:
        print("\n*** RESULT: INCONCLUSIVE ***")
    
    output = {
        'k_values': k_values,
        'n_seeds': n_seeds,
        'n_samples': n_samples,
        'epochs': epochs,
        'h1_constant': h1_constant,
        'h1_avg_residual': h1_avg_residual,
        'h2_formula': '1.944 - 0.623/k',
        'h2_avg_residual': h2_avg_residual,
        'results': {str(k): {
            'mean_sigma_star': results[k]['mean_sigma_star'],
            'sigma_k': results[k]['sigma_k'],
            'sigma_k_std': results[k]['sigma_k_std'],
            'n_valid': results[k]['n_valid']
        } for k in k_values}
    }
    
    with open('experiment_new_d_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to experiment_new_d_results.json")
    
    return output


if __name__ == "__main__":
    main()
