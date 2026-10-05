from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: matmul-dot
# Expected outcome: compile/prove candidate
# Precondition: Declare shared matrix/vector dimensions explicitly; when not expressible as @requires, encode with matching symbolic annotation names and keep as review obligation.
import numpy as np

@zk_circuit
def next_pythagorean_triples(previous: Integer[Array, 'N<=1024 M<=1024']):
    matrices = np.array([[-1, 2, 2], [-2, 1, 2], [-2, 2, 3], [1, 2, 2], [2, 1, 2], [2, 2, 3], [1, -2, 2], [2, -1, 2], [2, -2, 3]])
    next_triples = np.transpose(np.dot(matrices, np.transpose(previous)))
    next_triples = next_triples.reshape((3 * previous.shape[0], previous.shape[1]))
    return next_triples
