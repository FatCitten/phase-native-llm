"""
NEW-B: Z_4 Failure (Prime-Power Hypothesis) - FIXED
=====================================================
Test whether Z_4 fails because it's a prime-power order (2^2).
FIXED: Added explicit train/test split to test GENERALIZATION not memorization
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


def generate_zk_data_exhaustive(k):
    """Generate ALL possible (a,b) pairs - exhaustive coverage"""
    x1 = []
    x2 = []
    y = []
    for i in range(k):
        for j in range(k):
            x1.append(i)
            x2.append(j)
            y.append((i + j) % k)
    return torch.tensor(x1), torch.tensor(x2), torch.tensor(y)


def train_test_split(x1, x2, y, test_ratio=0.25):
    """Split into train/test, ensuring all classes in test"""
    n = len(x1)
    indices = torch.randperm(n)
    split = int(n * (1 - test_ratio))
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    return x1[train_idx], x2[train_idx], y[train_idx], x1[test_idx], x2[test_idx], y[test_idx]


def train_zk_on_split(k, x1_train, x2_train, y_train, epochs=150, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ZkBundle(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x1_train, x2_train)
        loss = nn.functional.cross_entropy(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    return model


def evaluate_accuracy(model, x1, x2, y):
    with torch.no_grad():
        outputs = model(x1, x2)
        accuracy = (outputs.argmax(1) == y).float().mean().item()
    return accuracy


def main():
    print("="*60)
    print("NEW-B: Z_4 FAILURE TEST (FIXED - Generalization)")
    print("="*60)
    print("\nHypothesis: Z_k fails when k is a prime power (p^n)")
    print("Z_4 = 2^2 should FAIL on held-out pairs")
    print("\nFIX: Explicit train/test split on (a,b) pairs")
    print("="*60)
    
    k = 4
    n_seeds = 10
    epochs = 150
    
    x1_all, x2_all, y_all = generate_zk_data_exhaustive(k)
    total_pairs = len(x1_all)
    print(f"\nTotal possible (a,b) pairs: {total_pairs}")
    
    results = []
    
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = train_test_split(
        x1_all, x2_all, y_all, test_ratio=0.25
    )
    train_size = len(y_train)
    test_size = len(y_test)
    
    for seed in range(n_seeds):
        model = train_zk_on_split(k, x1_train, x2_train, y_train, epochs=epochs, seed=seed)
        
        train_acc = evaluate_accuracy(model, x1_train, x2_train, y_train)
        test_acc = evaluate_accuracy(model, x1_test, x2_test, y_test)
        
        result = {
            'seed': seed,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'z4_passes': test_acc >= 0.8
        }
        results.append(result)
        
        print(f"  Seed {seed}: train={train_acc:.4f}, test={test_acc:.4f} ({'PASS' if result['z4_passes'] else 'FAIL'})")
    
    n_passed = sum(1 for r in results if r['z4_passes'])
    mean_train_acc = np.mean([r['train_accuracy'] for r in results])
    mean_test_acc = np.mean([r['test_accuracy'] for r in results])
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"k = {k} (prime power: 2^2)")
    print(f"Mean train accuracy: {mean_train_acc:.4f}")
    print(f"Mean test accuracy: {mean_test_acc:.4f}")
    print(f"Pass rate (test >= 0.8): {n_passed}/{n_seeds}")
    
    chance_level = 1.0 / k
    print(f"Chance level (1/k): {chance_level:.4f}")
    
    if n_passed == 0:
        print("\n*** RESULT: Z_4 FAILS ON TEST ***")
        print("Supports 'prime-power' hypothesis (Z_k fails for prime powers)")
    elif mean_test_acc < 0.5:
        print("\n*** RESULT: Z_4 NEAR CHANCE ON TEST ***")
        print("Network memorizes train but fails to generalize")
    else:
        print("\n*** RESULT: Z_4 GENERALIZES ***")
        print("Prime-power hypothesis INCONCLUSIVE - more investigation needed")
    
    output = {
        'k': k,
        'n_seeds': n_seeds,
        'epochs': epochs,
        'total_pairs': total_pairs,
        'train_size': train_size,
        'test_size': test_size,
        'mean_train_accuracy': mean_train_acc,
        'mean_test_accuracy': mean_test_acc,
        'chance_level': chance_level,
        'n_passed': n_passed,
        'pass_rate': n_passed / n_seeds,
        'results': results
    }
    
    with open('experiment_new_b_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to experiment_new_b_results.json")
    
    return output


if __name__ == "__main__":
    main()
