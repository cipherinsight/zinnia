# Source: NPBench scattering_self_energies (scattering_self_energies_numpy.py)
# Original signature: scattering_self_energies(neigh_idx, dH, G, D, Sigma) — see shapes from initialize().
# Migration notes:
#   - All loop bounds (Nkz, NE, Nqz, Nw, NA, NB, N3D, Norb) hoisted to module-level constants.
#   - Values from the "S" preset, kept small as-is (already <= 8 except NA=6, fits the <=32 budget).
#   - Original arrays are complex; migrated as Float (Zinnia has no complex). Indices remain Integer.
from zinnia import *

NKZ = 2
NE = 4
NQZ = 2
NW = 2
N3D = 2
NA = 6
NB = 2
NORB = 3


@zk_circuit
def scattering_self_energies(neigh_idx: NDArray[Integer, 6, 2],
                             dH: NDArray[Float, 6, 2, 2, 3, 3],
                             G: NDArray[Float, 2, 4, 6, 3, 3],
                             D: NDArray[Float, 2, 2, 6, 2, 2, 2],
                             Sigma: NDArray[Float, 2, 4, 6, 3, 3]):

    for k in range(G.shape[0]):
        for E in range(G.shape[1]):
            for q in range(D.shape[0]):
                for w in range(D.shape[1]):
                    for i in range(D.shape[-2]):
                        for j in range(D.shape[-1]):
                            for a in range(neigh_idx.shape[0]):
                                for b in range(neigh_idx.shape[1]):
                                    if E - w >= 0:
                                        dHG = G[k, E - w,
                                                neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD
