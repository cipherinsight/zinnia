# pythran program emin.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def Eandg(rOH, thetaHOH):
    """"
    Arguments: (internal coordinates of the water molecule)

    rOH            O-H bond distance
    thetaHOH       H-O-H bond angle


    Returns:

    E              Molecular force field energy
    grOH           O-H bond stretch gradient
    grthetaHOH     H-O-H bond angle bend gradient 

    
    Force field parameters:

    kOH            Harmonic force constant, O-H bond strech
    rOHe           Equilibrium distance, O-H
    kHOH           Harmonic angle bend force constant, H-O-H angle bend
    thetaHOHe      Equilibrium angle, H-O-H

    """
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
        E, grOH, gthetaHOH = Eandg(rOH, thetaHOH)
        if abs(grOH) > 0.001 / c or abs(gthetaHOH) > 0.01 / c:
            rOH = rOH - c * grOH
            thetaHOH = thetaHOH - c * gthetaHOH
    converged = abs(grOH) > 0.001 / c or abs(gthetaHOH) > 0.01 / c
    return (converged, E, rOH, thetaHOH)
