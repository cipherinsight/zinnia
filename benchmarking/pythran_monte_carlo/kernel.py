from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import math

@zk_circuit
@requires(lambda xmin, xmax, numSteps, rand, randsize: numSteps != 0)
@requires(lambda xmin, xmax, numSteps, rand, randsize: randsize != 0)
def montecarlo_integration(xmin: Float, xmax: Float, numSteps: Integer, rand: Float[Array, 'N<=1024'], randsize: Integer):

    @zk_chip
    def f(x: Float):
        return math.sin(x)
    ymin = f(xmin)
    ymax = ymin
    for i in range(numSteps):
        x = xmin + (xmax - xmin) * float(i) / numSteps
        y = f(x)
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    rectArea = (xmax - xmin) * (ymax - ymin)
    numPoints = numSteps
    ctr = 0
    for j in range(numPoints):
        x = xmin + (xmax - xmin) * rand[j % randsize]
        y = ymin + (ymax - ymin) * rand[j % randsize]
        if math.fabs(y) <= math.fabs(f(x)):
            if f(x) > 0 and y > 0 and (y <= f(x)):
                ctr += 1
            if f(x) < 0 and y < 0 and (y >= f(x)):
                ctr -= 1
    fnArea = rectArea * float(ctr) / numPoints
    return fnArea
