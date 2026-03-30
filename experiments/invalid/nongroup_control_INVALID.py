"""
NEW-C: Non-Group Control (max mod k)
====================================
Train on max(a,b) mod k - a non-group operation.
Expected: Phases should be LOW/random - no geometric emergence.
This confirms structure is group-specific.
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
    
    def forward(self, x1, x2):
        p1 = self.input_phases[x1]
        p2 = self.input_phases[x2]
        phi = (p1 + p2) % (2 * math.pi)
        dists = torch.abs(phi.unsqueeze(-1) - self.output_phases.unsqueeze(0))
        dists = dists % (2 * math.pi)
        dists = torch.min(dists, 2 * math.pi - dists)
        return -dists


def generate_max_mod_k_data(k, n_samples):
    x1 = torch.randint(0, k, (n_samples,))
    x2 = torch.randint(0, k, (n_samples,))
    y = torch.max(x1, x2) % k
    return x1, x2, y


def train_zk_on_operation(operation_fn, k, n_samples, epochs=150, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x1, x2, y = operation_fn(k, n_samples)
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model, x1, x2, y


def evaluate_accuracy(model, x1, x2, y):
    with torch.no_grad():
        outputs = model(x1, x2)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    return accuracy


def measure_phase_uniformity(phases, k):
    sorted_phases = torch.sort(phases % (2 * math.pi)).values
    diffs = (sorted_phases[1:] - sorted_phases[:-1]).abs()
    expected_spacing = 2 * math.pi / k
    uniformity = 1 - diffs.std().mean().item() / expected_spacing
    return uniformity


def main():
    print("="*60)
    print("NEW-C: NON-GROUP CONTROL (max mod k)")
    print("="*60)
    print("\nTrain on max(a,b) mod k (non-group operation)")
    print("Expected: Low phase uniformity (random), poor accuracy")
    print("="*60)
    
    k_values = [3, 5, 7, 11]
    n_seeds = 10
    n_samples = 1000
    epochs = 150
    
    results = {}
    
    for k in k_values:
        print(f"\n--- k = {k} ---")
        k_results = []
        
        for seed in range(n_seeds):
            model, x1, x2, y = train_zk_on_operation(
                generate_max_mod_k_data, k, n_samples, epochs=epochs, seed=seed
            )
            accuracy = evaluate_accuracy(model, x1, x2, y)
            
            input_uniformity = measure_phase_uniformity(model.input_phases.detach(), k)
            output_uniformity = measure_phase_uniformity(model.output_phases.detach(), k)
            
            result = {
                'seed': seed,
                'accuracy': accuracy,
                'input_phase_uniformity': input_uniformity,
                'output_phase_uniformity': output_uniformity
            }
            k_results.append(result)
            
            print(f"  Seed {seed}: acc={accuracy:.4f}, in_unif={input_uniformity:.4f}, out_unif={output_uniformity:.4f}")
        
        mean_accuracy = np.mean([r['accuracy'] for r in k_results])
        mean_input_uniformity = np.mean([r['input_phase_uniformity'] for r in k_results])
        mean_output_uniformity = np.mean([r['output_phase_uniformity'] for r in k_results])
        
        results[k] = {
            'mean_accuracy': mean_accuracy,
            'mean_input_uniformity': mean_input_uniformity,
            'mean_output_uniformity': mean_output_uniformity,
            'seeds': k_results
        }
        
        print(f"  Mean: acc={mean_accuracy:.4f}, in_unif={mean_input_uniformity:.4f}, out_unif={mean_output_uniformity:.4f}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for k in k_values:
        r = results[k]
        print(f"k={k}: accuracy={r['mean_accuracy']:.4f}, input_unif={r['mean_input_uniformity']:.4f}, output_unif={r['mean_output_uniformity']:.4f}")
    
    all_accuracies = [results[k]['mean_accuracy'] for k in k_values]
    all_uniformities = [results[k]['mean_output_uniformity'] for k in k_values]
    
    print(f"\nOverall mean accuracy: {np.mean(all_accuracies):.4f}")
    print(f"Overall mean output uniformity: {np.mean(all_uniformities):.4f}")
    
    if np.mean(all_accuracies) < 0.3 and np.mean(all_uniformities) < 0.5:
        print("\n*** RESULT: CONTROL CONFIRMED ***")
        print("Non-group operation shows poor accuracy and low phase uniformity")
    else:
        print("\n*** RESULT: UNEXPECTED - some structure learned ***")
    
    output = {
        'k_values': k_values,
        'n_seeds': n_seeds,
        'n_samples': n_samples,
        'epochs': epochs,
        'operation': 'max(a,b) mod k',
        'results': results
    }
    
    with open('experiment_new_c_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to experiment_new_c_results.json")
    
    return output


if __name__ == "__main__":
    main()
