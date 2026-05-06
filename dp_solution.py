"""Dynamic programming solution for the recurrence

    Psi_N(X) = sum_{i=1}^{X-N+1} Psi_{N-1}(X-i)

This implementation uses a bottom-up table with prefix sums so each row is
computed in linear time.

The default base case follows the common interpretation of the recurrence as
counting ways to write X as a sum of N positive integers:

    Psi_1(X) = 1  for X >= 1
    Psi_N(X) = 0  if X < N or N < 1 or X < 1
    Psi_N(N) = 1  for N >= 1
"""

from __future__ import annotations

from math import comb

"""Return Psi_N(X) using dynamic programming.

    Args:
        N: Recursion depth / number of parts.
        X: Target value.

    Returns:
        The value of Psi_N(X).
    """
def psi(N: int, X: int) -> int:
    if N < 1 or X < 1 or X < N:
        return 0
    if N == 1:
        return 1
    if N == X:
        return 1

    psi_n_minus_1 = [0] * (X + 1)
    for x in range(1, X + 1):
        psi_n_minus_1[x] = 1

    psi_n = [0] * (X + 1)
    for n in range(2, N + 1):
        psi_n = [0] * (X + 1)
        for x in range(n, X + 1):
            psi_n[x] = psi_n[x - 1] + psi_n_minus_1[x - 1]
        psi_n_minus_1 = psi_n

    return psi_n[X]


if __name__ == "__main__":
    samples = [(1, 5), (2, 5), (3, 7), (4, 10), (12,800)]
    for N, X in samples:
        value = psi(N, X)
        print(f"Psi_{N}({X}) = {value}")
