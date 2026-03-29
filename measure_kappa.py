"""
File 2: measure_kappa.py
=========================
Geometric measurement tools for Phase-Native Intelligence.

Provides:
  - compute_kappa(): Contextuality index from arxiv.org/pdf/2509.10536
  - compute_holonomy(): Measure holonomy around loops
  - compute_phase_coherence(): Measure phase alignment across fibers
  - Visualization tools for bundle geometry
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


def compute_kappa(
    fiber_phases: torch.Tensor,
    connection_field: Optional[torch.Tensor] = None,
    n_loops: int = 100
) -> float:
    """
    Compute contextuality index kappa.
    
    From arxiv.org/pdf/2509.10536: kappa measures non-trivial holonomy
    in the network. kappa > 0 indicates geometric structure is present.
    
    kappa = mean(||holonomy_matrices - I||_F) across sampled loops
    
    Args:
        fiber_phases: Tensor of shape (n_fibers,) or (batch, n_fibers)
                    Phase angles on U(1) for each fiber
        connection_field: Optional connection parameters
        n_loops: Number of random loops to sample
    
    Returns:
        float: Contextuality index (0 = flat/positional, >0 = geometric)
    """
    if fiber_phases.dim() == 1:
        fiber_phases = fiber_phases.unsqueeze(0)
    
    batch_size, n_fibers = fiber_phases.shape
    kappa_values = []
    
    for b in range(batch_size):
        phases = fiber_phases[b]
        
        # Sample random loops through the fiber bundle
        for _ in range(n_loops):
            # Random loop: start at random fiber, traverse random path, return
            start_idx = torch.randint(0, n_fibers, (1,)).item()
            
            # Compute holonomy around loop
            # For U(1) fibers: holonomy = sum of phase differences along path
            holonomy = torch.tensor(0.0)
            
            # Simple loop: go to next fiber and back
            end_idx = (start_idx + 1) % n_fibers
            
            # Phase difference between fibers
            delta_phi = phases[end_idx] - phases[start_idx]
            holonomy = delta_phi
            
            # Deviation from identity (flat space)
            # ||R - I|| where R = e^(i*holonomy)
            R_real = torch.cos(holonomy)
            R_imag = torch.sin(holonomy)
            deviation = torch.sqrt((R_real - 1) ** 2 + R_imag ** 2)
            kappa_values.append(deviation.item())
    
    return float(np.mean(kappa_values))


def compute_pairwise_holonomy(
    fiber_phases: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise holonomy (phase offsets) between all fiber pairs.
    
    Args:
        fiber_phases: Tensor of shape (n_fibers,) or (batch, n_fibers)
    
    Returns:
        Tensor of shape (n_fibers, n_fibers) with phase differences
        Entry (i, j) = e^(i * (phase_i - phase_j))
    """
    if fiber_phases.dim() == 1:
        fiber_phases = fiber_phases.unsqueeze(0)
    
    batch_size, n_fibers = fiber_phases.shape
    
    # Compute pairwise differences: phi_i - phi_j
    # Shape: (batch, n_fibers, 1) - (batch, 1, n_fibers) = (batch, n_fibers, n_fibers)
    delta_phases = fiber_phases.unsqueeze(-1) - fiber_phases.unsqueeze(1)
    
    # Convert to complex holonomy values
    holonomy_matrix = torch.complex(
        torch.cos(delta_phases),
        torch.sin(delta_phases)
    )
    
    return holonomy_matrix


def compute_phase_coherence(
    fiber_phases: torch.Tensor,
) -> float:
    """
    Measure overall coherence of phase distribution across fibers.
    
    High coherence = phases are aligned (weak geometric structure)
    Low coherence = phases are spread out (strong geometric structure)
    
    Args:
        fiber_phases: Tensor of shape (n_fibers,) or (batch, n_fibers)
    
    Returns:
        float: Coherence measure in [0, 1]
               1 = all phases aligned (flat)
               0 = phases uniformly distributed (maximal curvature)
    """
    if fiber_phases.dim() == 1:
        fiber_phases = fiber_phases.unsqueeze(0)
    
    batch_size, n_fibers = fiber_phases.shape
    
    # Mean resultant vector length (circular statistics)
    # R = |mean(e^(i*phases))|
    # R = 1: perfect alignment
    # R = 0: uniform distribution
    
    mean_vector = torch.mean(
        torch.complex(torch.cos(fiber_phases), torch.sin(fiber_phases)),
        dim=1
    )
    R = torch.abs(mean_vector)
    
    return float(torch.mean(R).item())


def compute_curvature_tensor(
    fiber_phases: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate local curvature at each fiber.
    
    Curvature is computed from second derivatives of phase field.
    
    Args:
        fiber_phases: Tensor of shape (n_fibers,)
    
    Returns:
        Tensor of shape (n_fibers,) with curvature estimates
    """
    n = len(fiber_phases)
    
    # Simple discrete curvature: second difference
    # kappa_i = phi_{i+1} - 2*phi_i + phi_{i-1}
    curvatures = []
    for i in range(n):
        left = fiber_phases[(i - 1) % n]
        center = fiber_phases[i]
        right = fiber_phases[(i + 1) % n]
        curvature = right - 2 * center + left
        curvatures.append(curvature.item())
    
    return torch.tensor(curvatures)


def measure_bundle_geometry(
    fiber_phases: torch.Tensor,
    connection_field: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Comprehensive geometric measurement of a fiber bundle.
    
    Args:
        fiber_phases: Phase angles for each fiber
        connection_field: Optional connection parameters
    
    Returns:
        Dictionary with geometric measurements:
            - kappa: contextuality index
            - coherence: phase coherence measure
            - curvature_mean: mean curvature
            - curvature_std: curvature variation
            - holonomy_trace: trace of holonomy matrix
    """
    if fiber_phases.dim() == 1:
        fiber_phases = fiber_phases.unsqueeze(0)
    
    results = {}
    
    # Kappa (contextuality index)
    results['kappa'] = compute_kappa(fiber_phases, connection_field)
    
    # Phase coherence
    results['coherence'] = compute_phase_coherence(fiber_phases)
    
    # Curvature
    curvatures = compute_curvature_tensor(fiber_phases[0])
    results['curvature_mean'] = float(torch.mean(torch.abs(curvatures)).item())
    results['curvature_std'] = float(torch.std(curvatures).item())
    
    # Holonomy matrix trace
    holonomy_matrix = compute_pairwise_holonomy(fiber_phases)
    results['holonomy_trace'] = float(torch.real(torch.trace(holonomy_matrix[0])).item())
    
    return results


def plot_phase_circle(
    fiber_phases: torch.Tensor,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot fibers on the U(1) circle.
    
    Args:
        fiber_phases: Phase angles for each fiber
        labels: Optional labels for each fiber
        save_path: Optional path to save figure
    """
    if fiber_phases.dim() == 1:
        fiber_phases = fiber_phases.unsqueeze(0)
    
    n_fibers = fiber_phases.shape[1]
    phases = fiber_phases[0].numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3, linewidth=2)
    
    # Fibers as points on circle
    x = np.cos(phases)
    y = np.sin(phases)
    
    for i in range(n_fibers):
        ax.scatter(x[i], y[i], s=200, zorder=5)
        
        if labels is not None:
            ax.annotate(labels[i], (x[i], y[i]), 
                       textcoords="offset points", 
                       xytext=(10, 10),
                       fontsize=12)
        else:
            ax.annotate(f'F{i}', (x[i], y[i]),
                       textcoords="offset points",
                       xytext=(10, 10),
                       fontsize=10)
        
        # Line from origin to point
        ax.plot([0, x[i]], [0, y[i]], 'b-', alpha=0.5)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Fiber Bundle on U(1) Circle')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase circle to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_holonomy_matrix(
    fiber_phases: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Visualize the pairwise holonomy matrix as a heatmap.
    
    Args:
        fiber_phases: Phase angles for each fiber
        save_path: Optional path to save figure
    """
    holonomy_matrix = compute_pairwise_holonomy(fiber_phases)
    
    # Take real part (cosine of phase difference)
    # This shows alignment: 1 = aligned, -1 = opposite
    real_parts = torch.real(holonomy_matrix[0]).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Real part heatmap
    im1 = axes[0].imshow(real_parts, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Holonomy Matrix (Real Part)')
    axes[0].set_xlabel('Fiber Index')
    axes[0].set_ylabel('Fiber Index')
    plt.colorbar(im1, ax=axes[0], label='cos(delta_phi)')
    
    # Add values to cells
    n = real_parts.shape[0]
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f'{real_parts[i, j]:.2f}',
                        ha='center', va='center', fontsize=8)
    
    # Phase differences as angular distances
    phases = fiber_phases[0].numpy()
    phase_diffs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = abs(phases[i] - phases[j]) % (2 * np.pi)
            phase_diffs[i, j] = min(diff, 2 * np.pi - diff)
    
    im2 = axes[1].imshow(phase_diffs, cmap='viridis', vmin=0, vmax=np.pi)
    axes[1].set_title('Phase Distance Matrix')
    axes[1].set_xlabel('Fiber Index')
    axes[1].set_ylabel('Fiber Index')
    plt.colorbar(im2, ax=axes[1], label='Distance (radians)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved holonomy matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_kappa_vs_training(
    kappa_history: List[float],
    loss_history: List[float],
    save_path: Optional[str] = None
):
    """
    Plot kappa (geometric structure) vs training progress.
    
    Shows how geometric structure develops as training proceeds.
    
    Args:
        kappa_history: List of kappa values over training
        loss_history: List of loss values over training
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Kappa over training
    axes[0].plot(kappa_history, 'purple', linewidth=2)
    axes[0].set_ylabel('Kappa (Contextuality Index)')
    axes[0].set_title('Geometric Structure Development During Training')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Flat space')
    
    # Loss over training
    axes[1].plot(loss_history, 'blue', linewidth=2)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved kappa plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_bundle(fiber_phases: torch.Tensor) -> str:
    """
    Generate a text analysis of bundle geometry.
    
    Args:
        fiber_phases: Phase angles for each fiber
    
    Returns:
        String with geometric analysis
    """
    geometry = measure_bundle_geometry(fiber_phases)
    
    analysis = []
    analysis.append("=" * 50)
    analysis.append("BUNDLE GEOMETRY ANALYSIS")
    analysis.append("=" * 50)
    analysis.append(f"\nContextuality Index (kappa): {geometry['kappa']:.4f}")
    analysis.append(f"  - kappa = 0: flat space (positional encoding)")
    analysis.append(f"  - kappa > 0: geometric structure present")
    
    analysis.append(f"\nPhase Coherence: {geometry['coherence']:.4f}")
    analysis.append(f"  - coherence = 1: all phases aligned")
    analysis.append(f"  - coherence = 0: phases uniformly distributed")
    
    analysis.append(f"\nCurvature:")
    analysis.append(f"  - Mean: {geometry['curvature_mean']:.4f}")
    analysis.append(f"  - Std:  {geometry['curvature_std']:.4f}")
    
    analysis.append(f"\nHolonomy Trace: {geometry['holonomy_trace']:.4f}")
    analysis.append(f"  - Trace = n: identity (flat)")
    analysis.append(f"  - Trace < n: non-trivial curvature")
    
    # Interpretation
    if geometry['kappa'] > 0.1:
        analysis.append("\nINTERPRETATION: Strong geometric structure detected.")
        analysis.append("The bundle has learned non-trivial curvature.")
    elif geometry['kappa'] > 0.01:
        analysis.append("\nINTERPRETATION: Moderate geometric structure.")
        analysis.append("Some curvature has developed.")
    else:
        analysis.append("\nINTERPRETATION: Minimal geometric structure.")
        analysis.append("The bundle is close to flat positional encoding.")
    
    analysis.append("=" * 50)
    
    return "\n".join(analysis)


class BundleAnalyzer:
    """Interactive bundle geometry analyzer."""
    
    def __init__(self):
        self.history = {
            'kappa': [],
            'coherence': [],
            'curvature': [],
            'steps': []
        }
    
    def update(self, fiber_phases: torch.Tensor, step: int):
        """Update analyzer with new measurement."""
        geometry = measure_bundle_geometry(fiber_phases)
        self.history['kappa'].append(geometry['kappa'])
        self.history['coherence'].append(geometry['coherence'])
        self.history['curvature'].append(geometry['curvature_mean'])
        self.history['steps'].append(step)
    
    def plot(self, save_path: Optional[str] = None):
        """Plot geometric evolution."""
        if len(self.history['steps']) == 0:
            print("No data to plot. Run update() first.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        steps = self.history['steps']
        
        axes[0].plot(steps, self.history['kappa'], 'purple', linewidth=2)
        axes[0].set_ylabel('Kappa')
        axes[0].set_title('Bundle Geometry Over Training')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(steps, self.history['coherence'], 'blue', linewidth=2)
        axes[1].set_ylabel('Coherence')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(steps, self.history['curvature'], 'red', linewidth=2)
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Curvature')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved geometry plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def summary(self) -> str:
        """Print summary statistics."""
        if len(self.history['steps']) == 0:
            return "No data collected."
        
        summary = []
        summary.append("\n" + "=" * 50)
        summary.append("BUNDLE ANALYSIS SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Total measurements: {len(self.history['steps'])}")
        summary.append(f"Final kappa: {self.history['kappa'][-1]:.4f}")
        summary.append(f"Max kappa: {max(self.history['kappa']):.4f}")
        summary.append(f"Final coherence: {self.history['coherence'][-1]:.4f}")
        summary.append(f"Final curvature: {self.history['curvature'][-1]:.4f}")
        summary.append("=" * 50)
        
        return "\n".join(summary)


def demo():
    """Demonstrate kappa measurement on simple examples."""
    print("=" * 60)
    print("MEASURE_KAPPA.PY — Demonstration")
    print("=" * 60)
    
    # Example 1: Flat bundle (all phases aligned)
    print("\n1. Flat bundle (aligned phases):")
    flat_phases = torch.tensor([0.0, 0.0, 0.0, 0.0])
    geometry = measure_bundle_geometry(flat_phases.unsqueeze(0))
    print(f"   kappa = {geometry['kappa']:.4f}")
    print(f"   coherence = {geometry['coherence']:.4f}")
    
    # Example 2: Uniform bundle (phases spread out)
    print("\n2. Uniform bundle (phases at 0, pi/2, pi, 3pi/2):")
    uniform_phases = torch.tensor([0.0, math.pi/2, math.pi, 3*math.pi/2])
    geometry = measure_bundle_geometry(uniform_phases.unsqueeze(0))
    print(f"   kappa = {geometry['kappa']:.4f}")
    print(f"   coherence = {geometry['coherence']:.4f}")
    
    # Example 3: XOR-like bundle (two groups)
    print("\n3. XOR-like bundle (two groups at 0 and pi):")
    xor_phases = torch.tensor([0.0, 0.0, math.pi, math.pi])
    geometry = measure_bundle_geometry(xor_phases.unsqueeze(0))
    print(f"   kappa = {geometry['kappa']:.4f}")
    print(f"   coherence = {geometry['coherence']:.4f}")
    
    # Example 4: Random bundle
    print("\n4. Random bundle (uniform random phases):")
    random_phases = torch.rand(8) * 2 * math.pi
    geometry = measure_bundle_geometry(random_phases.unsqueeze(0))
    print(f"   kappa = {geometry['kappa']:.4f}")
    print(f"   coherence = {geometry['coherence']:.4f}")
    
    # Analysis
    print("\n" + analyze_bundle(xor_phases.unsqueeze(0)))
    
    print("\n" + "=" * 60)
    print("Demonstration complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
