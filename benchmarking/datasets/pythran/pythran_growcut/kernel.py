from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import math
import numpy as np

@zk_chip
def window_floor(idx: Float, radius: Float):
    if radius > idx:
        return 0
    else:
        return idx - radius

@zk_chip
def window_ceil(idx: Float, ceil: Float, radius: Float):
    if idx + radius > ceil:
        return ceil
    else:
        return idx + radius

@zk_chip
def python_kernel(image: Float[Array, 'N<=1024 M<=1024 K<=1024'], state: Float[Array, 'N<=1024 M<=1024 K<=1024'], state_next: Float[Array, 'N<=1024 M<=1024 K<=1024'], window_radius: Float):
    changes = 0
    sqrt_3 = math.sqrt(3.0)
    height = image.shape[0]
    width = image.shape[1]
    for j in range(width):
        for i in range(height):
            winning_colony = state[i, j, 0]
            defense_strength = state[i, j, 1]
            for jj in range(window_floor(j, window_radius), window_ceil(j + 1, width, window_radius)):
                for ii in range(window_floor(i, window_radius), window_ceil(i + 1, height, window_radius)):
                    if ii == i and jj == j:
                        continue
                    d = image[i, j, 0] - image[ii, jj, 0]
                    s = d * d
                    for k in range(1, 3):
                        d = image[i, j, k] - image[ii, jj, k]
                        s += d * d
                    gval = 1.0 - math.sqrt(s) / sqrt_3
                    attack_strength = gval * state[ii, jj, 1]
                    if attack_strength > defense_strength:
                        defense_strength = attack_strength
                        winning_colony = state[ii, jj, 0]
                        changes += 1
            state_next[i, j, 0] = winning_colony
            state_next[i, j, 1] = defense_strength
    return changes

@zk_circuit
@requires(lambda N: 0 <= N <= 1024)
def test(N: Integer):
    image = np.zeros((N, N, 3))
    state = np.zeros((N, N, 2))
    state_next = np.empty_like(state)
    state[0, 0, 0] = 1
    state[0, 0, 1] = 1
    return python_kernel(image, state, state_next, 10)
