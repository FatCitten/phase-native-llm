"""
Experiment A: Data Independence Test
====================================
Vary training set size n across [64, 128, 256, 512, 1024, 2048].
For each (n, k), run full sigma* detection with 10 seeds.
Question: Does sigma* change with n?
Null hypothesis: sigma* = C/k regardless of n.
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
    
    def get_phases(self):
        return self.input_phases.detach().cpu().numpy()


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


def evaluate_with_noise(model, x1, x2, y, noise_sigma):
    with torch.no_grad():
        outputs = model.forward_with_noise(x1, x2, noise_sigma)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    return accuracy


def find_critical_sigma(model, x1, x2, y, sigma_range=(0.0, 0.5), n_points=20):
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], n_points)
    for sigma in sigma_values:
        acc = evaluate_with_noise(model, x1, x2, y, sigma)
        if acc < 0.8:
            return sigma, acc
    return None, 1.0


def main():
    print("="*60)
    print("EXPERIMENT A: DATA INDEPENDENCE TEST")
    print("="*60)
    
    n_values = [64, 128, 256, 512, 1024, 2048]
    k_values = [3, 5, 7, 11]
    n_seeds = 10
    
    results = {}
    
    for n in n_values:
        print(f"\n=== n = {n} ===")
        results[n] = {}
        
        for k in k_values:
            print(f"\n  k = {k}")
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
            
            sk = mean_sigma * k if mean_sigma else None
            if mean_sigma:
                print(f"    sigma* = {mean_sigma:.4f}, sigma* x k = {mean_sigma * k:.4f}")
            else:
                print(f"    sigma* = N/A")
    
    print("\n" + "="*60)
    print("SUMMARY TABLE: sigma* vs n")
    print("="*60)
    print("\n| n    | k=3    | k=5    | k=7    | k=11   |")
    print("|------|--------|--------|--------|--------|")
    
    for n in n_values:
        row = f"| {n:4d} |"
        for k in k_values:
            s = results[n][k]['mean_sigma_star']
            if s:
                row += f" {s:6.4f} |"
            else:
                row += "   N/A  |"
        print(row)
    
    print("\n" + "="*60)
    print("SUMMARY TABLE: sigma* × k vs n (should be ~1.82)")
    print("="*60)
    print("\n| n    | k=3    | k=5    | k=7    | k=11   | Mean   |")
    print("|------|--------|--------|--------|--------|--------|")
    
    row_means = []
    for n in n_values:
        row = f"| {n:4d} |"
        for k in k_values:
            sk = results[n][k]['sigma_k']
            if sk:
                row += f" {sk:6.4f} |"
            else:
                row += "   N/A  |"
            row_means.append(sk)
        valid = [x for x in row_means if x is not None]
        row += f" {np.mean(valid):6.4f} |" if valid else "   N/A  |"
        print(row)
        row_means = []
    
    all_ratios = []
    for n in n_values:
        for k in k_values:
            if results[n][k]['sigma_k']:
                all_ratios.append(results[n][k]['sigma_k'])
    
    print(f"\nOverall mean sigma* × k: {np.mean(all_ratios):.4f}")
    print(f"Overall std sigma* × k: {np.std(all_ratios):.4f}")
    
    if np.std(all_ratios) / np.mean(all_ratios) < 0.1:
        print("\n*** DATA INDEPENDENT CONFIRMED ***")
        print("sigma* is independent of training set size n!")
    else:
        print("\n*** DATA INDEPENDENCE NOT CONFIRMED ***")
    
    with open('experiment_a_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiment_a_results.json")
    
    return results


if __name__ == "__main__":
    main()
