"""
Experiment 3: Input Phase Structure at Different Training Noise
=============================================================
Train at different σ values, measure INPUT weight structure.
Does the LEARNED SOLUTION change shape when training noise crosses σ*?
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


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
    
    def get_input_phases(self):
        return self.input_phases.detach().cpu().numpy()
    
    def get_output_phases(self):
        return self.output_phases.detach().cpu().numpy()


def generate_zk_data(k, n_samples):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = (x1 + x2) % k
    return x1, x2, y


def train_zk_with_noise(k, n_samples, epochs, seed, train_noise):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1, x2, y = generate_zk_data(k, n_samples)
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        if train_noise > 0:
            outputs = model.forward_with_noise(x1, x2, train_noise)
        else:
            outputs = model(x1, x2)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model


def measure_input_structure(input_phases):
    """Measure structure of input phases"""
    phases = np.sort(input_phases % (2 * math.pi))
    k = len(phases)
    
    diffs = np.diff(phases)
    diffs = np.append(diffs, phases[0] + 2*np.pi - phases[-1])
    
    return {
        'phases': phases,
        'diffs': diffs,
        'mean_diff': np.mean(diffs),
        'std_diff': np.std(diffs),
        'max_diff': np.max(diffs),
        'min_diff': np.min(diffs),
        'uniformity': 1.0 - np.std(diffs) / (np.mean(diffs) + 1e-10)
    }


def main():
    print("="*60)
    print("EXPERIMENT 3: INPUT STRUCTURE vs TRAINING NOISE")
    print("="*60)
    
    k = 7
    train_noise_values = [0.0, 0.1, 0.2, 0.26, 0.3, 0.4, 0.5]
    n_seeds = 15
    n_samples = 1000
    epochs = 150
    
    results = {}
    
    print(f"\nTraining k={k} at various training noise levels...")
    print("-" * 60)
    
    for train_noise in train_noise_values:
        print(f"\ntrain_noise = {train_noise:.2f}")
        
        input_structures = []
        output_structures = []
        final_losses = []
        
        for seed in range(n_seeds):
            model = train_zk_with_noise(k, n_samples, epochs, seed, train_noise)
            
            input_phases = model.get_input_phases()
            output_phases = model.get_output_phases()
            
            input_struct = measure_input_structure(input_phases)
            output_struct = measure_input_structure(output_phases)
            
            input_structures.append(input_struct)
            output_structures.append(output_struct)
            
            with torch.no_grad():
                x1, x2, y = generate_zk_data(k, 500)
                outputs = model(x1, x2)
                loss = nn.functional.cross_entropy(outputs, y)
                final_losses.append(loss.item())
        
        results[train_noise] = {
            'input_uniformity': [s['uniformity'] for s in input_structures],
            'input_std_diff': [s['std_diff'] for s in input_structures],
            'output_uniformity': [s['uniformity'] for s in output_structures],
            'output_std_diff': [s['std_diff'] for s in output_structures],
            'final_loss': final_losses,
            'input_phases': [s['phases'].tolist() for s in input_structures],
            'output_phases': [s['phases'].tolist() for s in output_structures]
        }
        
        print(f"  Input uniformity: {np.mean([s['uniformity'] for s in input_structures]):.4f} +/- {np.std([s['uniformity'] for s in input_structures]):.4f}")
        print(f"  Output uniformity: {np.mean([s['uniformity'] for s in output_structures]):.4f} +/- {np.std([s['uniformity'] for s in output_structures]):.4f}")
        print(f"  Final loss: {np.mean(final_losses):.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"\n{'train_noise':>12} | {'in_uniform':>10} | {'out_uniform':>10} | {'loss':>10}")
    print("-" * 50)
    for train_noise in train_noise_values:
        r = results[train_noise]
        in_u = np.mean(r['input_uniformity'])
        out_u = np.mean(r['output_uniformity'])
        loss = np.mean(r['final_loss'])
        print(f"{train_noise:12.2f} | {in_u:10.4f} | {out_u:10.4f} | {loss:10.4f}")
    
    train_noise_plot = list(results.keys())
    in_uniform_means = [np.mean(results[n]['input_uniformity']) for n in train_noise_plot]
    in_uniform_stds = [np.std(results[n]['input_uniformity']) for n in train_noise_plot]
    out_uniform_means = [np.mean(results[n]['output_uniformity']) for n in train_noise_plot]
    out_uniform_stds = [np.std(results[n]['output_uniformity']) for n in train_noise_plot]
    loss_means = [np.mean(results[n]['final_loss']) for n in train_noise_plot]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    ax.errorbar(train_noise_plot, in_uniform_means, yerr=in_uniform_stds, fmt='o-', capsize=3, color='steelblue', linewidth=2, markersize=8)
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* = 0.26')
    ax.set_xlabel('Training Noise (sigma)', fontsize=12)
    ax.set_ylabel('Input Phase Uniformity', fontsize=12)
    ax.set_title('Input Uniformity vs Training Noise', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.errorbar(train_noise_plot, out_uniform_means, yerr=out_uniform_stds, fmt='s-', capsize=3, color='forestgreen', linewidth=2, markersize=8)
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* = 0.26')
    ax.set_xlabel('Training Noise (sigma)', fontsize=12)
    ax.set_ylabel('Output Phase Uniformity', fontsize=12)
    ax.set_title('Output Uniformity vs Training Noise', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(train_noise_plot, loss_means, 'o-', color='coral', linewidth=2, markersize=8)
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* = 0.26')
    ax.set_xlabel('Training Noise (sigma)', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Loss vs Training Noise', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for i, train_noise in enumerate([0.0, 0.26, 0.5]):
        if train_noise in results:
            phases = results[train_noise]['input_phases'][0]
            ax.hist(phases, bins=20, alpha=0.5, label=f'sigma={train_noise}')
    ax.set_xlabel('Input Phase (radians)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Input Phase Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for i, train_noise in enumerate([0.0, 0.26, 0.5]):
        if train_noise in results:
            diffs = results[train_noise]['input_std_diff']
            ax.scatter([train_noise]*len(diffs), diffs, alpha=0.6, s=50)
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7, label='sigma* = 0.26')
    ax.set_xlabel('Training Noise (sigma)', fontsize=12)
    ax.set_ylabel('Input Phase Std Dev', fontsize=12)
    ax.set_title('Input Phase Variability', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax2 = ax.twinx()
    l1 = ax.plot(train_noise_plot, in_uniform_means, 'o-', color='steelblue', linewidth=2, markersize=8, label='Input Uniformity')
    l2 = ax2.plot(train_noise_plot, loss_means, 's--', color='coral', linewidth=2, markersize=8, label='Loss')
    ax.axvline(x=0.26, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Training Noise (sigma)', fontsize=12)
    ax.set_ylabel('Input Uniformity', fontsize=12, color='steelblue')
    ax2.set_ylabel('Loss', fontsize=12, color='coral')
    ax.set_title('Structure vs Performance', fontsize=14, fontweight='bold')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_3_input_structure.png', dpi=300, bbox_inches='tight')
    print("\nSaved to experiment_3_input_structure.png")
    
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)
    
    u_below = np.mean(results[0.2]['input_uniformity'])
    u_above = np.mean(results[0.3]['input_uniformity'])
    change = abs(u_below - u_above)
    
    print(f"Input uniformity at sigma=0.2: {u_below:.4f}")
    print(f"Input uniformity at sigma=0.3: {u_above:.4f}")
    print(f"Change: {change:.4f}")
    
    if change < 0.05:
        print("\n*** INPUT STRUCTURE IS ROBUST ACROSS sigma* ***")
        print("The learned representation maintains structure")
        print("even as training noise increases.")
    else:
        print("\n*** INPUT STRUCTURE CHANGES AT sigma* ***")
    
    save_data = {str(k): v for k, v in results.items()}
    with open('experiment_3_input_structure.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\nSaved to experiment_3_input_structure.json")


if __name__ == "__main__":
    main()
