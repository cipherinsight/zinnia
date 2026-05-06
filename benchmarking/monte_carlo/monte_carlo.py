# Source: Pythran tests/cases/monte_carlo.py
# Original #pythran export: montecarlo_integration(float, float, int, float list, int)
from zinnia import *
import math


@zk_circuit
def montecarlo_integration(xmin: float, xmax: float, numSteps: int, rand: NDArray[Float, 64], randsize: int):
    def f(x):
        _zinnia_result = math.sin(x)

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
            if f(x) > 0 and y > 0 and y <= f(x):
                ctr += 1
            if f(x) < 0 and y < 0 and y >= f(x):
                ctr -= 1

    fnArea = rectArea * float(ctr) / numPoints
    _zinnia_result = fnArea
