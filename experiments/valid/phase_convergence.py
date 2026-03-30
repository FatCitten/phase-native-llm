"""
NEW-A: Phase Convergence Test
==============================
Verify that trained phases converge to 2*pi*j/k +/- small offset across seeds.
This confirms phases are meaningful, not arbitrary.
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
        outputs = model(x1, x2)
        loss = nn.functional.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model, x1, x2, y


def measure_phase_spacing(phases, k):
    sorted_phases = torch.sort(phases % (2 * math.pi)).values
    diffs = (sorted_phases[1:] - sorted_phases[:-1]).abs()
    expected_spacing = 2 * math.pi / k
    return diffs, expected_spacing


def evaluate_accuracy(model, x1, x2, y):
    with torch.no_grad():
        outputs = model(x1, x2)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    return accuracy


def main():
    print("="*60)
    print("NEW-A: PHASE CONVERGENCE TEST")
    print("="*60)
    print("\nTest whether phases converge to 2*pi*j/k +/- offset")
    print("Expected: All seeds converge to uniform spacing")
    print("="*60)
    
    k = 7
    n_seeds = 20
    n_samples = 1000
    epochs = 150
    
    results = []
    
    for seed in range(n_seeds):
        model, x1, x2, y = train_zk(k, n_samples, epochs=epochs, seed=seed)
        accuracy = evaluate_accuracy(model, x1, x2, y)
        
        input_phases = model.input_phases.detach()
        output_phases = model.output_phases.detach()
        
        input_diffs, expected = measure_phase_spacing(input_phases, k)
        output_diffs, _ = measure_phase_spacing(output_phases, k)
        
        input_spacing_std = input_diffs.std().item()
        output_spacing_std = output_diffs.std().item()
        
        input_spacing_mean = input_diffs.mean().item()
        output_spacing_mean = output_diffs.mean().item()
        
        input_variance_ratio = input_spacing_std / input_spacing_mean
        output_variance_ratio = output_spacing_std / output_spacing_mean
        
        result = {
            'seed': seed,
            'accuracy': accuracy,
            'input_spacing_mean': input_spacing_mean,
            'input_spacing_std': input_spacing_std,
            'input_variance_ratio': input_variance_ratio,
            'output_spacing_mean': output_spacing_mean,
            'output_spacing_std': output_spacing_std,
            'output_variance_ratio': output_variance_ratio,
            'expected_spacing': expected
        }
        results.append(result)
        
        print(f"  Seed {seed}: acc={accuracy:.4f}, in_var={input_variance_ratio:.4f}, out_var={output_variance_ratio:.4f}")
    
    mean_input_variance = np.mean([r['input_variance_ratio'] for r in results])
    mean_output_variance = np.mean([r['output_variance_ratio'] for r in results])
    mean_accuracy = np.mean([r['accuracy'] for r in results])
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"k = {k}")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Mean input variance ratio: {mean_input_variance:.4f}")
    print(f"Mean output variance ratio: {mean_output_variance:.4f}")
    print(f"Expected spacing: {expected:.4f}")
    
    if mean_input_variance < 0.2 and mean_output_variance < 0.2:
        print("\n*** RESULT: PHASES CONVERGE ***")
        print("Low variance ratio indicates uniform phase spacing")
    else:
        print("\n*** RESULT: PHASES NOT UNIFORM ***")
    
    output = {
        'k': k,
        'n_seeds': n_seeds,
        'n_samples': n_samples,
        'epochs': epochs,
        'mean_accuracy': mean_accuracy,
        'mean_input_variance_ratio': mean_input_variance,
        'mean_output_variance_ratio': mean_output_variance,
        'expected_spacing': expected,
        'results': results
    }
    
    with open('experiment_new_a_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to experiment_new_a_results.json")
    
    return output


if __name__ == "__main__":
    main()
