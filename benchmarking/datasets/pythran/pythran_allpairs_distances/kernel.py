from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.
import numpy as np

@zk_chip
def dists(X: Float[Array, 'N<=1024'], Y: Float[Array, 'N<=1024']):
    return np.array([[np.sum((x - y) ** 2) for x in X] for y in Y])

@zk_circuit
def allpairs_distances(d: Integer):
    X = np.arange(600 * d).reshape((600, d))
    Y = np.arange(200 * d).reshape((200, d))
    return dists(X, Y)
