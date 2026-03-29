"""
File 0: sanity_checks.py
========================
Environment and formula verification BEFORE building File 1.
Tests:
  1. Complex number operations work correctly
  2. Holonomy formula gives correct output on hand-calculated values
  3. Lambda schedule works mechanically
  4. PyTorch version and CUDA availability
"""

import math
import torch
import numpy as np


def get_lambda(epoch: int) -> float:
    """Lambda schedule for holonomy loss weight."""
    if epoch < 20:
        return 0.0
    if epoch < 50:
        return 0.1
    if epoch < 100:
        return 0.3
    return 0.1


def compute_holonomy_loss(R_actual: torch.Tensor, R_predicted: torch.Tensor) -> torch.Tensor:
    """
    Holonomy loss: ||R_actual - R_predicted||²_F
    For U(1) = complex number, this is |z1 - z2|²
    """
    return torch.abs(R_actual - R_predicted).pow(2)


def test_pytorch_version():
    """Verify PyTorch >= 2.0 with complex tensor support."""
    version = torch.__version__
    major, minor = map(int, version.split('.')[:2])
    assert major > 2 or (major == 2 and minor >= 0), \
        f"PyTorch 2.0+ required, got {version}"
    print(f"[PASS] PyTorch version: {version}")


def test_complex_tensor_support():
    """Verify complex tensor creation and basic operations."""
    # Create complex number: e^(i * π) = -1 + 0i
    phi = torch.tensor(math.pi)
    z = torch.complex(torch.cos(phi), torch.sin(phi))
    
    # Should equal -1 + 0i
    assert torch.allclose(z.real, torch.tensor(-1.0), atol=1e-5), \
        f"Real part should be -1.0, got {z.real}"
    assert torch.allclose(z.imag, torch.tensor(0.0), atol=1e-5), \
        f"Imag part should be 0.0, got {z.imag}"
    print(f"[PASS] Complex tensor: e^(i*pi) = {z}")


def test_holonomy_at_zero():
    """If R_actual == R_predicted, loss should = 0."""
    R_actual = torch.complex(torch.tensor(-1.0), torch.tensor(0.0))
    R_predicted = torch.complex(torch.tensor(-1.0), torch.tensor(0.0))
    loss = compute_holonomy_loss(R_actual, R_predicted)
    
    assert loss.item() < 1e-10, \
        f"Loss should be ~0 for matching holonomies, got {loss.item()}"
    print(f"[PASS] Zero mismatch: loss = {loss.item():.2e}")


def test_holonomy_at_max_mismatch():
    """If R_actual = +1, R_predicted = -1, loss should = 4.0."""
    R_actual = torch.complex(torch.tensor(1.0), torch.tensor(0.0))
    R_predicted = torch.complex(torch.tensor(-1.0), torch.tensor(0.0))
    loss = compute_holonomy_loss(R_actual, R_predicted)
    
    assert torch.allclose(loss, torch.tensor(4.0), atol=1e-5), \
        f"Max mismatch loss should be 4.0, got {loss.item()}"
    print(f"[PASS] Max mismatch: loss = {loss.item():.2f} (expected 4.0)")


def test_holonomy_at_quarter_turn():
    """If R_actual = +1, R_predicted = i, loss should = 2.0."""
    R_actual = torch.complex(torch.tensor(1.0), torch.tensor(0.0))
    R_predicted = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
    loss = compute_holonomy_loss(R_actual, R_predicted)
    
    # |1 - i|² = (1-0)² + (0-1)² = 1 + 1 = 2
    assert torch.allclose(loss, torch.tensor(2.0), atol=1e-5), \
        f"Quarter-turn mismatch should be 2.0, got {loss.item()}"
    print(f"[PASS] Quarter-turn mismatch: loss = {loss.item():.2f} (expected 2.0)")


def test_xor_holonomy_expected():
    """Test the expected holonomy values for XOR."""
    # XOR truth table:
    # (0,0) -> 0: φ_A=0, φ_B=0, holonomy=0
    # (0,1) -> 1: φ_A=0, φ_B=π, holonomy=π
    # (1,0) -> 1: φ_A=π, φ_B=0, holonomy=π
    # (1,1) -> 0: φ_A=π, φ_B=π, holonomy=0
    
    def expected_holonomy(input_A, input_B):
        """Compute expected holonomy for XOR input."""
        phi_A = input_A * math.pi
        phi_B = input_B * math.pi
        return phi_A - phi_B
    
    # Verify expected values
    assert abs(expected_holonomy(0, 0)) < 1e-5, "(0,0) holonomy should be 0"
    assert abs(expected_holonomy(1, 1)) < 1e-5, "(1,1) holonomy should be 0"
    assert abs(abs(expected_holonomy(0, 1)) - math.pi) < 1e-5, "(0,1) holonomy should be π"
    assert abs(abs(expected_holonomy(1, 0)) - math.pi) < 1e-5, "(1,0) holonomy should be π"
    
    print("[PASS] XOR holonomy expectations:")
    print(f"       (0,0) -> holonomy = {expected_holonomy(0,0):.2f} (target: 0)")
    print(f"       (0,1) -> holonomy = {expected_holonomy(0,1):.2f} (target: pi)")
    print(f"       (1,0) -> holonomy = {expected_holonomy(1,0):.2f} (target: pi)")
    print(f"       (1,1) -> holonomy = {expected_holonomy(1,1):.2f} (target: 0)")


def test_lambda_schedule():
    """Verify lambda schedule is correct."""
    test_cases = [
        (0,   0.0),
        (10,  0.0),
        (19,  0.0),
        (20,  0.1),
        (25,  0.1),
        (49,  0.1),
        (50,  0.3),
        (75,  0.3),
        (99,  0.3),
        (100, 0.1),
        (200, 0.1),
    ]
    
    for epoch, expected in test_cases:
        actual = get_lambda(epoch)
        assert actual == expected, \
            f"Epoch {epoch}: expected λ={expected}, got λ={actual}"
    
    print("[PASS] Lambda schedule:")
    print("       Epochs  0-19: lambda = 0.0")
    print("       Epochs 20-49: lambda = 0.1")
    print("       Epochs 50-99: lambda = 0.3")
    print("       Epoch  100+ : lambda = 0.1")


def test_cuda_info():
    """Print CUDA availability and GPU info."""
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU 0: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"[INFO] Compute capability: {props.major}.{props.minor}")


def test_numpy_version():
    """Verify NumPy version."""
    version = np.__version__
    major = int(version.split('.')[0])
    assert major >= 1, f"NumPy 1.0+ required, got {version}"
    print(f"[PASS] NumPy version: {version}")


def test_matplotlib_import():
    """Verify matplotlib is available."""
    try:
        import matplotlib
        print(f"[PASS] Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        raise AssertionError("Matplotlib not installed. Run: pip install matplotlib")


def test_tqdm_import():
    """Verify tqdm is available."""
    try:
        import tqdm
        print(f"[PASS] tqdm installed")
    except ImportError:
        raise AssertionError("tqdm not installed. Run: pip install tqdm")


def main():
    """Run all sanity checks."""
    print("=" * 60)
    print("PHASE-NATIVE LLM — SANITY CHECKS")
    print("=" * 60)
    print()
    
    # Environment checks
    print("[ENVIRONMENT]")
    test_pytorch_version()
    test_numpy_version()
    test_matplotlib_import()
    test_tqdm_import()
    test_cuda_info()
    print()
    
    # Formula checks
    print("[COMPLEX TENSOR OPERATIONS]")
    test_complex_tensor_support()
    print()
    
    print("[HOLONOMY LOSS FORMULA]")
    test_holonomy_at_zero()
    test_holonomy_at_max_mismatch()
    test_holonomy_at_quarter_turn()
    test_xor_holonomy_expected()
    print()
    
    print("[LAMBDA SCHEDULE]")
    test_lambda_schedule()
    print()
    
    # Summary
    print("=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("SAFE TO BUILD FILE 1: minimal_bundle.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
