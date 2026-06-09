from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.

@zk_circuit
def Eandg(rOH: Float, thetaHOH: Float):
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
