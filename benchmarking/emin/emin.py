# Source: Pythran tests/cases/emin.py
# Original #pythran export: run()
from zinnia import *


@zk_chip
def Eandg(rOH, thetaHOH) -> Tuple[Float, Float, Float]:
    kOH = 50.0
    rOHe = 0.95
    kHOH = 50.0
    thetaHOHe = 104.5

    E = 2 * kOH * (rOH - rOHe) ** 2 + kHOH * (thetaHOH - thetaHOHe) ** 2
    grOH = 2 * kOH * (rOH - rOHe)
    grthetaHOH = 2 * kHOH * (thetaHOH - thetaHOHe)

    return (E, grOH, grthetaHOH)


@zk_circuit
def run():
    c = 0.005
    n_steps = 1000000

    rOH = 10.0
    thetaHOH = 180.0

    for i in range(n_steps):
        (E, grOH, gthetaHOH) = Eandg(rOH, thetaHOH)
        if (abs(grOH) > 0.001 / c or abs(gthetaHOH) > 0.01 / c):
            rOH = rOH - c * grOH
            thetaHOH = thetaHOH - c * gthetaHOH

    converged = (abs(grOH) > 0.001 / c or abs(gthetaHOH) > 0.01 / c)

    _zinnia_result = converged, E, rOH, thetaHOH
