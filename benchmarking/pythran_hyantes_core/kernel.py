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

@zk_circuit
@requires(lambda xmin, ymin, xmax, ymax, step, range_, range_x, range_y, t: step != 0.0)
@requires(lambda xmin, ymin, xmax, ymax, step, range_, range_x, range_y, t: range_ != 0.0)
def run(xmin: Float, ymin: Float, xmax: Float, ymax: Float, step: Float, range_: Float, range_x: Integer, range_y: Integer, t: Float[Array, 'N<=1024 M<=1024']):
    pt = [[0] * range_y for _ in range(range_x)]
    'omp parallel for'
    for i in range(range_x):
        for j in range(range_y):
            s = 0
            for k in t:
                tmp = 6368.0 * math.acos(math.cos(xmin + step * i) * math.cos(k[0]) * math.cos(ymin + step * j - k[1]) + math.sin(xmin + step * i) * math.sin(k[0]))
                if tmp < range_:
                    s += k[2] / (1 + tmp)
            pt[i][j] = s
    return pt
