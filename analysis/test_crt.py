"""
Verify CRT function with unit tests
"""

def crt(a1, m1, a2, m2):
    """Solve x ≡ a1 (mod m1), x ≡ a2 (mod m2)"""
    inv_m1 = pow(m1, -1, m2)  # modular inverse of m1 mod m2
    inv_m2 = pow(m2, -1, m1)  # modular inverse of m2 mod m1
    M = m1 * m2
    return (a1 * m2 * inv_m2 + a2 * m1 * inv_m1) % M


# Test cases
tests = [
    (0, 2, 2, 3, 2),   # 2%2=0, 2%3=2
    (0, 2, 1, 3, 4),   # 4%2=0, 4%3=1
    (1, 2, 0, 3, 3),   # 3%2=1, 3%3=0
    (1, 2, 1, 3, 1),   # 1%2=1, 1%3=1
    (0, 3, 0, 5, 0),   # 0%3=0, 0%5=0
    (1, 3, 2, 5, 7),   # 7%3=1, 7%5=2
    (2, 3, 4, 5, 14),  # 14%3=2, 14%5=4
]

print("Testing CRT function...")
all_pass = True
for a1, m1, a2, m2, expected in tests:
    result = crt(a1, m1, a2, m2)
    status = "PASS" if result == expected else "FAIL"
    if result != expected:
        all_pass = False
    print(f"  crt({a1}, {m1}, {a2}, {m2}) = {result}, expected {expected} [{status}]")

if all_pass:
    print("\n*** ALL TESTS PASSED ***")
else:
    print("\n*** SOME TESTS FAILED ***")
