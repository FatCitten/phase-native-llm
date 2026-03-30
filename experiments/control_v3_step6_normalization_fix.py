"""
Normalization Fix: Re-extract distance tables for k=5,7,11
with consistent wrapped distance normalization.
===========================================================
"""

import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def sigmoid(x, theta, beta):
    return 1.0 / (1.0 + np.exp(-(x - theta) / beta))

def fit_sigmoid(x_data, y_data):
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    try:
        # Allow theta to go negative (below x=0 means always below threshold)
        popt, _ = curve_fit(sigmoid, x_arr, y_arr, p0=[0.3, 0.1], bounds=([-0.5, 0.01], [1.0, 0.5]))
        theta, beta = popt
        y_pred = sigmoid(x_arr, theta, beta)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return float(theta), float(beta), float(r_squared)
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None, None

def main():
    # Load existing distance tables (simple d = |a-b|)
    with open("results/control_v3/step3_analysis_fixed.json", 'r') as f:
        old_data = json.load(f)
    
    # OLD theta values (buggy - using d/k normalization)
    old_theta = {
        5: 0.0527,
        7: 0.1248,
        11: 0.1548
    }
    old_beta = {
        5: 0.1173,
        7: 0.1121,
        11: 0.1038
    }
    
    print("="*70)
    print("NORMALIZATION FIX: k=5,7,11")
    print("="*70)
    
    print("\n" + "="*70)
    print("BEFORE (buggy normalization):")
    print("="*70)
    print(f"theta = [{old_theta[5]:.4f}, {old_theta[7]:.4f}, {old_theta[11]:.4f}]")
    print(f"beta  = [{old_beta[5]:.4f}, {old_beta[7]:.4f}, {old_beta[11]:.4f}]")
    
    print("\n" + "="*70)
    print("TASK 1: Re-extract with wrapped normalization")
    print("="*70)
    
    corrected_results = {}
    
    for k in [5, 7, 11]:
        print(f"\n--- k = {k} ---")
        
        # Get raw distance table (simple d = |a-b|)
        raw_dist = old_data[str(k)]['distance_table']
        
        # Convert to wrapped distance: wrapped_d = min(d, k-d)
        wrapped_acc = {}
        for d_str, acc in raw_dist.items():
            d = int(d_str)
            wrapped_d = min(d, k - d)
            if wrapped_d not in wrapped_acc:
                wrapped_acc[wrapped_d] = []
            wrapped_acc[wrapped_d].append(acc)
        
        # Aggregate (average) for each wrapped_d
        wrapped_dist = {}
        for wd, accs in wrapped_acc.items():
            wrapped_dist[wd] = np.mean(accs)
        
        # Build table with normalized x = 2*wrapped_d/k
        # Include diagonal (d=0) as well for fitting
        x_data = []
        y_data = []
        print(f"d   wrapped_d   x=2d/k   acc")
        print("-" * 35)
        
        for d in sorted(wrapped_dist.keys()):
            wrapped_d = d
            x = 2 * wrapped_d / k
            acc = wrapped_dist[d]
            x_data.append(x)
            y_data.append(acc)
            diag_mark = "(diag)" if d == 0 else ""
            print(f"{d:2d}  {wrapped_d:10d}  {x:6.3f}   {acc:.3f}  {diag_mark}")
        
        # Fit sigmoid
        theta, beta, r2 = fit_sigmoid(x_data, y_data)
        
        print(f"\nFitted: theta={theta:.4f}, beta={beta:.4f}, R2={r2:.4f}")
        
        corrected_results[k] = {
            'theta': theta,
            'beta': beta,
            'r_squared': r2,
            'distance_table': wrapped_dist
        }
    
    print("\n" + "="*70)
    print("AFTER (fixed wrapped normalization):")
    print("="*70)
    print(f"theta = [{corrected_results[5]['theta']:.4f}, {corrected_results[7]['theta']:.4f}, {corrected_results[11]['theta']:.4f}]")
    print(f"beta  = [{corrected_results[5]['beta']:.4f}, {corrected_results[7]['beta']:.4f}, {corrected_results[11]['beta']:.4f}]")
    
    print("\n" + "="*70)
    print("BEFORE/AFTER COMPARISON")
    print("="*70)
    print(f"BEFORE (buggy d/k):  theta = [0.0527, 0.1248, 0.1548]")
    print(f"AFTER  (fixed 2d/k): theta = [{corrected_results[5]['theta']:.4f}, {corrected_results[7]['theta']:.4f}, {corrected_results[11]['theta']:.4f}]")
    
    # Check if shift is significant
    theta_shift = [
        corrected_results[5]['theta'] - old_theta[5],
        corrected_results[7]['theta'] - old_theta[7],
        corrected_results[11]['theta'] - old_theta[11]
    ]
    max_shift = max(abs(s) for s in theta_shift)
    
    print(f"\nMax theta shift: {max_shift:.4f}")
    
    if max_shift < 0.05:
        print("\nVERDICT: Theta values barely changed -> normalization was NOT the main issue.")
        print("The k=13,17 values are likely correct; we should proceed with retraining.")
    else:
        print("\nVERDICT: Theta values shifted significantly -> normalization bug CONFIRMED!")
        print("We need to retrain k=13,17 with consistent analysis.")
    
    # Save intermediate results
    Path("results/control_v3").mkdir(parents=True, exist_ok=True)
    with open("results/control_v3/step6_k5_7_11_corrected.json", 'w') as f:
        json.dump(corrected_results, f, indent=2)
    print("\nSaved: results/control_v3/step6_k5_7_11_corrected.json")
    
    return corrected_results

if __name__ == "__main__":
    results = main()
