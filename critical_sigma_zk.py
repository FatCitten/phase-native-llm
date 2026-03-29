"""
Find critical sigma (σ*) for Z_k modular arithmetic - Simplified version
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


def train_and_evaluate_zk(k, noise_level=0.0, n_samples=2000, epochs=200, seed=42):
    """Train Z_k with input noise (flip probability)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1, x2, y = generate_zk_data(k, n_samples)
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        # Apply noise: flip bits with probability noise_level
        if noise_level > 0:
            flip_mask1 = torch.rand_like(x1.float()) < noise_level
            flip_mask2 = torch.rand_like(x2.float()) < noise_level
            x1_noisy = torch.where(flip_mask1, k - 1 - x1, x1)
            x2_noisy = torch.where(flip_mask2, k - 1 - x2, x2)
        else:
            x1_noisy, x2_noisy = x1, x2
        
        optimizer.zero_grad()
        outputs = model(x1_noisy, x2_noisy)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        outputs = model(x1, x2)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    
    return accuracy


def find_critical_noise_for_k(k):
    """Find critical flip probability where accuracy drops."""
    print(f"\n--- k={k} ---")
    
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for noise in noise_levels:
        accs = []
        for seed in range(5):
            acc = train_and_evaluate_zk(k, noise_level=noise, seed=seed)
            accs.append(acc)
        mean_acc = np.mean(accs)
        results.append((noise, mean_acc))
        print(f"  noise={noise:.2f}: accuracy={mean_acc:.4f}")
        
        if mean_acc < 0.5 and noise > 0.1:
            break
    
    # Find where accuracy drops below 50%
    for noise, acc in results:
        if acc < 0.5:
            return noise, results
    
    return noise_levels[-1], results


def main():
    print("="*60)
    print("FINDING CRITICAL NOISE FOR Z_k")
    print("="*60)
    
    k_values = [3, 5, 7, 11]
    all_results = {}
    
    for k in k_values:
        critical_noise, results = find_critical_noise_for_k(k)
        
        # Find interpolated critical point (where acc = 0.5)
        for i in range(len(results) - 1):
            if results[i][1] > 0.5 and results[i+1][1] < 0.5:
                # Linear interpolation
                x1, y1 = results[i]
                x2, y2 = results[i+1]
                critical = x1 + (x2 - x1) * (0.5 - y1) / (y2 - y1)
                break
        else:
            critical = critical_noise
        
        pi_over_k = math.pi / k
        
        all_results[k] = {
            'critical_noise': critical,
            'pi_over_k': pi_over_k,
            'ratio': critical / pi_over_k
        }
        
        print(f"  ==> critical noise for k={k}: {critical:.4f}, pi/k={pi_over_k:.4f}")
        print(f"      ratio = {critical/pi_over_k:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n| k   | critical noise | pi/k   | ratio |")
    print("|------|----------------|---------|-------|")
    for k, r in all_results.items():
        print(f"| {k:2d}  |     {r['critical_noise']:.4f}   | {r['pi_over_k']:.4f} | {r['ratio']:.4f} |")
    
    # Check if ratio is constant
    ratios = [r['ratio'] for r in all_results.values()]
    print(f"\nRatio (critical/pi/k): mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")
    
    if np.std(ratios) / np.mean(ratios) < 0.3:
        print("\n==> CRITICAL NOISE IS PROPORTIONAL TO pi/k!")
    
    # Save
    with open('critical_noise_zk.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to critical_noise_zk.json")


if __name__ == "__main__":
    main()
