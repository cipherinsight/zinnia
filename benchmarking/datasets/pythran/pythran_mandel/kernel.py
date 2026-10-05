from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.

@zk_circuit
@requires(lambda size, x_center, y_center, max_iteration: 0 <= size <= 1024)
@requires(lambda size, x_center, y_center, max_iteration: size != 0)
def mandel(size: Integer, x_center: Float, y_center: Float, max_iteration: Integer):
    out = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        'omp parallel for'
        for j in range(size):
            x, y = (x_center + 4.0 * float(i - size / 2) / size, y_center + 4.0 * float(j - size / 2) / size)
            a, b = (0.0, 0.0)
            iteration = 0
            while a ** 2 + b ** 2 <= 4.0 and iteration < max_iteration:
                a, b = (a ** 2 - b ** 2 + x, 2 * a * b + y)
                iteration += 1
            if iteration == max_iteration:
                color_value = 255
            else:
                color_value = iteration * 10 % 255
            out[i][j] = color_value
    return out
