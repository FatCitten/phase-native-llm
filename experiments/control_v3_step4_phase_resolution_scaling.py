"""
Phase Resolution Scaling Analysis
=================================
TASK 1: Plot accuracy vs d/k (normalized) for k=5,7,11 on same axes
TASK 2: Fit sigmoid to each k and pooled data
TASK 3: Interpret critical angle and sharpness
TASK 4: Save structured results to JSON
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from pathlib import Path

# Hardcoded data from step3_analysis
DATA = {
    5: {0: 0.400, 1: 0.750, 2: 1.000, 3: 1.000, 4: 1.000},
    7: {0: 0.286, 1: 0.500, 2: 0.800, 3: 1.000, 4: 1.000, 5: 1.000, 6: 1.000},
    11: {0: 0.182, 1: 0.400, 2: 0.556, 3: 0.625, 4: 1.000, 
         5: 1.000, 6: 1.000, 7: 1.000, 8: 1.000, 9: 1.000, 10: 1.000},
}

def sigmoid(x, theta, beta):
    """acc(d/k) = 1 / (1 + exp(-(d/k - theta) / beta))"""
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-(x - theta) / beta))

def fit_sigmoid(x_data, y_data):
    """Fit sigmoid to data, return theta, beta, r_squared"""
    x_arr = np.asarray(x_data)
    y_arr = np.asarray(y_data)
    
    try:
        popt, _ = curve_fit(sigmoid, x_arr, y_arr, p0=[0.3, 0.05], bounds=([0, 0.001], [0.5, 0.5]))
        theta, beta = popt
        
        y_pred = sigmoid(x_arr, theta, beta)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return theta, beta, r_squared
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None, None

def main():
    k_values = [5, 7, 11]
    colors = {5: 'blue', 7: 'green', 11: 'red'}
    
    # Build normalized data
    all_data = {}
    for k in k_values:
        all_data[k] = {'d_over_k': [], 'acc': []}
        for d, acc in DATA[k].items():
            all_data[k]['d_over_k'].append(d / k)
            all_data[k]['acc'].append(acc)
    
    # Print normalized table
    print("=" * 60)
    print("NORMALIZED TABLE: accuracy vs d/k")
    print("=" * 60)
    print(f"{'d/k':>8}  {'k=5':>8}  {'k=7':>8}  {'k=11':>8}")
    print("-" * 60)
    
    max_dk = max(max(v['d_over_k']) for v in all_data.values())
    dk_points = np.arange(0, round(max_dk*20)+1) / 20
    
    for dk in dk_points:
        row = f"{dk:8.3f}"
        for k in k_values:
            matches = [all_data[k]['acc'][i] for i, x in enumerate(all_data[k]['d_over_k']) if abs(x - dk) < 0.001]
            if matches:
                row += f"  {matches[0]:7.3f}"
            else:
                row += f"  {'---':>7}"
        print(row)
    
    # Per-k sigmoid fits
    print("\n" + "=" * 60)
    print("PER-K SIGMOID FITS")
    print("=" * 60)
    
    per_k_results = {}
    for k in k_values:
        x = all_data[k]['d_over_k']
        y = all_data[k]['acc']
        theta, beta, r2 = fit_sigmoid(x, y)
        per_k_results[k] = {'theta': theta, 'beta': beta, 'r_squared': r2}
        print(f"k={k:2d}: theta = {theta:.4f}, beta = {beta:.4f}, R2 = {r2:.4f}")
    
    # Universal (pooled) sigmoid fit
    print("\n" + "=" * 60)
    print("UNIVERSAL SIGMOID FIT (pooled all k)")
    print("=" * 60)
    
    all_x = []
    all_y = []
    for k in k_values:
        all_x.extend(all_data[k]['d_over_k'])
        all_y.extend(all_data[k]['acc'])
    
    universal_theta, universal_beta, universal_r2 = fit_sigmoid(all_x, all_y)
    print(f"theta_universal = {universal_theta:.4f}")
    print(f"beta_universal = {universal_beta:.4f}")
    print(f"R2 = {universal_r2:.4f}")
    
    # Critical angle interpretation
    print("\n" + "=" * 60)
    print("CRITICAL ANGLE INTERPRETATION")
    print("=" * 60)
    
    for k in k_values:
        theta = per_k_results[k]['theta']
        angle_rad = theta * 2 * np.pi
        angle_deg = np.degrees(angle_rad)
        print(f"k={k:2d}: theta = {theta:.4f} -> {theta*2*math.pi:.4f} rad = {angle_deg:.2f} deg")
    
    print(f"\nUniversal: theta = {universal_theta:.4f} -> {universal_theta*2*math.pi:.4f} rad = {np.degrees(universal_theta*2*np.pi):.2f} deg")
    
    # Sharpness interpretation
    print("\n" + "=" * 60)
    print("SHARPNESS INTERPRETATION")
    print("=" * 60)
    
    avg_beta = universal_beta
    if avg_beta < 0.05:
        sharpness_verdict = "SHARP CLIFF (binary resolution limit)"
    elif avg_beta > 0.15:
        sharpness_verdict = "GRADUAL DEGRADATION (soft limit)"
    else:
        sharpness_verdict = "MODERATE TRANSITION"
    
    print(f"beta = {avg_beta:.4f}")
    print(f"Verdict: {sharpness_verdict}")
    
    # Collapse quality (MAD between k=7 and k=11 at shared d/k)
    print("\n" + "=" * 60)
    print("COLLAPSE QUALITY")
    print("=" * 60)
    
    k7_x = np.array(all_data[7]['d_over_k'])
    k11_x = np.array(all_data[11]['d_over_k'])
    k7_y = np.array(all_data[7]['acc'])
    k11_y = np.array(all_data[11]['acc'])
    
    common_x = np.intersect1d(np.round(k7_x, 3), np.round(k11_x, 3))
    
    if len(common_x) > 0:
        k7_interp = np.interp(common_x, k7_x, k7_y)
        k11_interp = np.interp(common_x, k11_x, k11_y)
        mad = np.mean(np.abs(k7_interp - k11_interp))
    else:
        mad = float('nan')
    
    if mad < 0.05:
        collapse_verdict = "STRONG COLLAPSE (universal behavior confirmed)"
    elif mad > 0.15:
        collapse_verdict = "WEAK COLLAPSE (k-dependent behavior)"
    else:
        collapse_verdict = "MODERATE COLLAPSE"
    
    print(f"MAD(k=7 vs k=11) = {mad:.4f}")
    print(f"Verdict: {collapse_verdict}")
    
    # Plot
    print("\n" + "=" * 60)
    print("GENERATING PLOT")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_fit = np.linspace(0, 0.5, 100)
    
    for k in k_values:
        x = all_data[k]['d_over_k']
        y = all_data[k]['acc']
        
        ax.scatter(x, y, color=colors[k], s=80, zorder=5, label=f'k={k} (data)')
        
        theta = per_k_results[k]['theta']
        beta = per_k_results[k]['beta']
        y_fit = sigmoid(x_fit, theta, beta)
        ax.plot(x_fit, y_fit, color=colors[k], linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='acc=0.5')
    ax.axvline(x=universal_theta, color='purple', linestyle=':', alpha=0.7, label=f'theta={universal_theta:.3f}')
    
    ax.set_xlabel('d/k (normalized distance from diagonal)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Phase Resolution Scaling: Accuracy vs Normalized Distance\n(theta={universal_theta:.3f}, beta={universal_beta:.3f}, R2={universal_r2:.3f})', fontsize=13)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    Path("results/control_v3").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/control_v3/phase_resolution_scaling.png", dpi=150, bbox_inches='tight')
    print("Saved: results/control_v3/phase_resolution_scaling.png")
    
    # Save JSON
    results = {
        "per_k": {
            str(k): {"theta": per_k_results[k]['theta'], "beta": per_k_results[k]['beta'], "r_squared": per_k_results[k]['r_squared']}
            for k in k_values
        },
        "universal": {
            "theta": universal_theta,
            "beta": universal_beta,
            "r_squared": universal_r2,
            "critical_angle_radians": universal_theta * 2 * np.pi,
            "critical_angle_degrees": np.degrees(universal_theta * 2 * np.pi)
        },
        "collapse_quality": {
            "mad_k7_vs_k11": mad,
            "verdict": collapse_verdict
        }
    }
    
    with open("results/control_v3/phase_resolution_fit.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: results/control_v3/phase_resolution_fit.json")
    
    return results

import math
if __name__ == "__main__":
    results = main()
