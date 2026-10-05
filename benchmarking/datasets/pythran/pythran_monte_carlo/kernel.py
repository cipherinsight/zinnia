# pythran program monte_carlo.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import math

@zk_circuit
@requires(lambda xmin, xmax, numSteps, rand, randsize: numSteps != 0)
@requires(lambda xmin, xmax, numSteps, rand, randsize: randsize != 0)
def montecarlo_integration(xmin: Float, xmax: Float, numSteps: Integer, rand: Float[Array, '100'], randsize: Integer):

    def f(x):
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
