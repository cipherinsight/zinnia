from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np
from math import ceil

@zk_chip
def physics(masspoints: Float[Array, 'N<=1024'], dt: Float, plunk: Float, which: Float):
    ppos = masspoints[1]
    cpos = masspoints[0]
    N = cpos.shape[0]
    HOOKE_K = 2100000.0
    DAMPING = 0.0001
    MASS = 0.01
    force = np.zeros((N, 2))
    for i in range(1, N):
        dx, dy = cpos[i] - cpos[i - 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        assert dist != 0
        fmag = -HOOKE_K * dist
        cosine = dx / dist
        sine = dy / dist
        fvec = np.array([fmag * cosine, fmag * sine])
        force[i - 1] -= fvec
        force[i] += fvec
    force[0] = force[-1] = (0, 0)
    force[which][1] += plunk
    accel = force / MASS
    npos = (2 - DAMPING) * cpos - (1 - DAMPING) * ppos + accel * dt ** 2
    masspoints[1] = cpos
    masspoints[0] = npos

@zk_circuit
def test(PARTICLE_COUNT: Integer):
    SUBDIVISION = 300
    FRAMERATE = 60
    count = PARTICLE_COUNT
    width, height = (600, 200)
    masspoints = np.empty((2, count, 2), np.float64)
    initpos = np.zeros(count, np.float64)
    for i in range(1, count):
        initpos[i] = initpos[i - 1] + float(width) / count
    masspoints[:, :, 0] = initpos
    masspoints[:, :, 1] = height / 2
    f = 15
    plunk_pos = count // 2
    physics(masspoints, 1.0 / (SUBDIVISION * FRAMERATE), f, plunk_pos)
    return masspoints[0, count // 2]
