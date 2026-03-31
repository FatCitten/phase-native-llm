import torch
import torch.nn as nn
import math
import json
import numpy as np
from pathlib import Path
import sys

DEVICE = torch.device('cpu')
K_VALUES = [11, 17, 23]

class ZkBundleExplicit_v2c(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        class_phases = 2 * math.pi * torch.arange(k, device=DEVICE) / k
        self.register_buffer('fourier_basis', torch.stack([
            torch.cos(class_phases), torch.sin(class_phases)
        ], dim=1))
    
    def forward(self, a, b):
        phases_a = 2 * math.pi * a.float() / self.k
        phases_b = 2 * math.pi * b.float() / self.k
        result_phase = phases_a + phases_b
        
        result_emb = torch.stack([
            torch.cos(result_phase),
            torch.sin(result_phase)
        ], dim=-1)
        
        logits = result_emb @ self.fourier_basis.T
        return logits

def make_data(k):
    a = torch.arange(k, device=DEVICE).repeat(k)
    b = torch.arange(k, device=DEVICE).repeat_interleave(k)
    target = (a + b) % k
    
    n = k * k
    indices = torch.randperm(n)
    train_idx = indices[:n // 2]
    test_idx = indices[n // 2:]
    
    return {
        'a_train': a[train_idx], 'b_train': b[train_idx], 'target_train': target[train_idx],
        'a_test': a[test_idx], 'b_test': b[test_idx], 'target_test': target[test_idx]
    }

def test_model(model_class, k, data):
    model = model_class(k).to(DEVICE)
    
    a_train = data['a_train']
    b_train = data['b_train']
    target_train = data['target_train']
    a_test = data['a_test']
    b_test = data['b_test']
    target_test = data['target_test']
    
    with torch.no_grad():
        train_logits = model(a_train, b_train)
        train_acc = (train_logits.argmax(dim=-1) == target_train).float().mean().item()
        
        test_logits = model(a_test, b_test)
        test_acc = (test_logits.argmax(dim=-1) == target_test).float().mean().item()
    
    return train_acc, test_acc

def main():
    print("="*60)
    print("ZkBundleExplicit_v2c - Fourier Readout")
    print("Testing EXACT mathematical solution at step 0")
    print("="*60)
    
    results = {}
    
    for k in K_VALUES:
        data = make_data(k)
        n_train = len(data['target_train'])
        n_test = len(data['target_test'])
        
        print(f"\nk={k} | train={n_train} | test={n_test}")
        
        train_acc, test_acc = test_model(ZkBundleExplicit_v2c, k, data)
        
        print(f"  Step 0 - train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
        
        if test_acc >= 0.99:
            print(f"  => EXACT SOLUTION!")
        elif test_acc >= 0.95:
            print(f"  => NEAR EXACT")
        else:
            print(f"  => FAILED")
        
        results[f'k{k}'] = {'train_acc': train_acc, 'test_acc': test_acc}
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for k in K_VALUES:
        acc = results[f'k{k}']['test_acc']
        status = "PASS" if acc >= 0.99 else "FAIL"
        if acc < 0.99:
            all_passed = False
        print(f"k={k}: {acc:.4f} [{status}]")
    
    if all_passed:
        print("\n>>> EXACT SOLUTION ACHIEVED - ZERO TRAINING NEEDED <<<")
        print("This proves the CONNECTION (angle addition) is mathematically exact!")
    
    Path('results').mkdir(exist_ok=True)
    with open('results/zkbundle_explicit_v2c.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/zkbundle_explicit_v2c.json")

if __name__ == '__main__':
    main()