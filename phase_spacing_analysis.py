"""
Phase Spacing Analysis
=======================
Analyze learned phase spacing in ZkBundle models to detect precision problems.
"""

import torch
import numpy as np
import json
import math
from pathlib import Path

def find_checkpoint(k, base_dir="results/control_v3/models"):
    """Find checkpoint for given k (prefer seed0, no hm suffix)"""
    import glob
    
    # Try pattern: k{k}_seed0.pt (no hm suffix = hm1)
    pattern1 = f"{base_dir}/k{k}_seed0.pt"
    matches = sorted(glob.glob(pattern1))
    if matches:
        return matches[0]
    
    # Try k{k}_hm1_seed0.pt
    pattern2 = f"{base_dir}/k{k}_hm1_seed0.pt"
    matches = sorted(glob.glob(pattern2))
    if matches:
        return matches[0]
    
    return None

def compute_phase_spacing(phases):
    """Compute spacing between adjacent sorted phases (with wraparound)."""
    phases_sorted = np.sort(phases % (2 * math.pi))
    n = len(phases_sorted)
    gaps = []
    for i in range(n):
        next_i = (i + 1) % n
        gap = (phases_sorted[next_i] - phases_sorted[i]) % (2 * math.pi)
        if gap < 1e-10:  # handle wraparound at 2π
            gap = (2 * math.pi - phases_sorted[i]) + phases_sorted[next_i]
        gaps.append(gap)
    return np.array(gaps)

def analyze_checkpoint(path, k):
    """Load checkpoint and analyze phase spacing."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # Get state dict (checkpoint structure: {seed, k, state_dict, ...})
    if 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    
    # Find output phases parameter
    output_phases = None
    for key in state.keys():
        if 'output' in key.lower() and 'bundle' in key.lower():
            output_phases = state[key].detach().numpy()
            print(f"  Found: {key}")
            break
    
    if output_phases is None:
        print(f"  WARNING: No output phases found in {path}")
        print(f"  Available keys: {list(state.keys())[:10]}")
    
    if output_phases is None:
        print(f"  WARNING: No output phases found in {path}")
        return None
    
    # Handle multiple bundles - analyze first bundle
    if output_phases.ndim > 1:
        output_phases = output_phases[0]
    
    print(f"  Shape: {output_phases.shape}, range: [{output_phases.min():.3f}, {output_phases.max():.3f}]")
    
    # Compute gaps
    gaps = compute_phase_spacing(output_phases)
    theoretical_gap = 2 * math.pi / k
    
    return {
        "min_gap": float(np.min(gaps)),
        "max_gap": float(np.max(gaps)),
        "mean_gap": float(np.mean(gaps)),
        "std_gap": float(np.std(gaps)),
        "theoretical_gap": theoretical_gap,
        "ratio_min_theory": float(np.min(gaps) / theoretical_gap),
        "ratio_mean_theory": float(np.mean(gaps) / theoretical_gap),
        "ratio_std_theory": float(np.std(gaps) / theoretical_gap),
        "cv": float(np.std(gaps) / np.mean(gaps)),  # coefficient of variation
        "phases": output_phases.tolist()
    }

# ============================================================================
# MAIN
# ============================================================================
print("=" * 70)
print("PHASE SPACING ANALYSIS")
print("=" * 70)

k_values = [5, 7, 11, 13, 17, 19, 21, 23, 29]

results = []
for k in k_values:
    print(f"\n--- k = {k} ---")
    ckpt_path = find_checkpoint(k)
    if ckpt_path is None:
        print(f"  No checkpoint found!")
        continue
    
    print(f"  Loading: {ckpt_path}")
    result = analyze_checkpoint(ckpt_path, k)
    result["k"] = k
    result["checkpoint"] = ckpt_path
    results.append(result)

# Print summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"\n{'k':>4} | {'min_gap':>10} | {'theor_2pi/k':>12} | {'ratio':>8} | {'CV':>8} | {'regime':>7}")
print("-" * 65)

for r in results:
    k = r["k"]
    regime = "II" if k <= 17 else "III"
    print(f"{k:>4} | {r['min_gap']:>10.4f} | {r['theoretical_gap']:>12.4f} | {r['ratio_min_theory']:>8.3f} | {r['cv']:>8.3f} | {regime:>7}")

# Save results
output = {
    "analysis": "Phase spacing in learned output phases",
    "results": results
}

Path("results/ceiling_decay").mkdir(parents=True, exist_ok=True)
with open("results/ceiling_decay/phase_spacing.json", 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: results/ceiling_decay/phase_spacing.json")

# Additional: check correlation between k and irregularity
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)
k_arr = np.array([r["k"] for r in results])
cv_arr = np.array([r["cv"] for r in results])
ratio_arr = np.array([r["ratio_min_theory"] for r in results])

# Pearson correlation
mean_k = np.mean(k_arr)
mean_cv = np.mean(cv_arr)
numer = np.sum((k_arr - mean_k) * (cv_arr - mean_cv))
denom = np.sqrt(np.sum((k_arr - mean_k)**2) * np.sum((cv_arr - mean_cv)**2))
pearson_cv = numer / denom if denom > 0 else 0

mean_ratio = np.mean(ratio_arr)
numer_r = np.sum((k_arr - mean_k) * (ratio_arr - mean_ratio))
denom_r = np.sqrt(np.sum((k_arr - mean_k)**2) * np.sum((ratio_arr - mean_ratio)**2))
pearson_ratio = numer_r / denom_r if denom_r > 0 else 0

print(f"  Pearson r (k vs CV): {pearson_cv:.4f}")
print(f"  Pearson r (k vs min_ratio): {pearson_ratio:.4f}")

if pearson_cv > 0.7:
    print("  => INCREASING IRREGULARITY with k (supports precision loss hypothesis)")
else:
    print("  => NO clear trend in irregularity with k")
