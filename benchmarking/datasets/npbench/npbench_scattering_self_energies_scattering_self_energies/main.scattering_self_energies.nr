// npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies
// (NPBench scattering_self_energies), Noir baseline.
// Kernel: for every k, E, q, w, i, j, a, b with E >= w,
// Sigma[k, E, a] += (G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i])
//                   @ (dH[a, b, j] * D[q, w, a, b, i, j])
// on complex NORB x NORB blocks, Sigma in place; the output is Sigma after the
// call (`mutated_outputs`).
// The two factors depend on fewer indices than the loop nest runs over, so
// each is computed once: the gathered blocks per (k, E', a, b), the left
// factor per (k, E', a, b, i), the right one per (q, w, a, b, i, j). Every
// scalar complex product rounds its four real products individually, then
// subtracts/adds. Matrix products sum those rounded complex products.
// The gather is native indexing at `index_pos` (NumPy wrap and bounds) of
// neigh_idx[a, b], once per (a, b); it pins neigh_idx[a, b] to a valid
// index, which implies its domain (an int64), so neigh_idx gets no separate
// domain check.
// @outputs: float[NKZ * NE * NA * BLOCK * 2]

use noir_hw::{check_inputs, index_pos, mul};

// Sizes (hw_check.sizes): from fixture.json.
global NA: u32 = 6; // = neigh_idx_0
global NB: u32 = 2; // = neigh_idx_1
global N3D: u32 = 2; // = dH_2
global NORB: u32 = 3; // = dH_3
global NKZ: u32 = 2; // = G_0
global NE: u32 = 4; // = G_1
global GA: u32 = 6; // = G_2
global NQZ: u32 = 2; // = D_0
global NW: u32 = 2; // = D_1
global BLOCK: u32 = NORB * NORB;
// `index_pos` width: 8 bits up to 128 elements, 28 above (NOIR_BASELINES.md).
global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (G_2 - 1) // 128)

// Complex matrix product, rounding each elementary real product.
fn complex_matmul(x: [[Field; 2]; BLOCK], y: [[Field; 2]; BLOCK]) -> [[Field; 2]; BLOCK] {
    let mut out: [[Field; 2]; BLOCK] = [[0; 2]; BLOCK];
    for r in 0..NORB {
        for c in 0..NORB {
            let mut re: Field = 0;
            let mut im: Field = 0;
            for t in 0..NORB {
                let u = x[r * NORB + t];
                let v = y[t * NORB + c];
                re += mul(u[0], v[0]) - mul(u[1], v[1]);
                im += mul(u[0], v[1]) + mul(u[1], v[0]);
            }
            out[r * NORB + c] = [re, im];
        }
    }
    out
}

fn main(
    neigh_idx: [[Field; NB]; NA],
    dH: [[[[[Field; 2]; BLOCK]; N3D]; NB]; NA],
    G: [[[[[Field; 2]; BLOCK]; GA]; NE]; NKZ],
    D: [[[[[[[Field; 2]; N3D]; N3D]; NB]; NA]; NW]; NQZ],
    Sigma: [[[[[Field; 2]; BLOCK]; NA]; NE]; NKZ],
) -> pub [[[[[Field; 2]; BLOCK]; NA]; NE]; NKZ] {
    // Input domain: every element of dH, G, D and Sigma is a Q32 code in the
    // window (neigh_idx: see the header).
    for a in 0..NA {
        for b in 0..NB {
            for i in 0..N3D {
                for t in 0..BLOCK {
                    check_inputs(dH[a][b][i][t]);
                }
            }
        }
    }
    for k in 0..NKZ {
        for e in 0..NE {
            for p in 0..GA {
                for t in 0..BLOCK {
                    check_inputs(G[k][e][p][t]);
                }
            }
            for a in 0..NA {
                for t in 0..BLOCK {
                    check_inputs(Sigma[k][e][a][t]);
                }
            }
        }
    }
    for qz in 0..NQZ {
        for w in 0..NW {
            for a in 0..NA {
                for b in 0..NB {
                    for i in 0..N3D {
                        for j in 0..N3D {
                            check_inputs(D[qz][w][a][b][i][j]);
                        }
                    }
                }
            }
        }
    }

    // G[k, E', neigh_idx[a, b]] @ dH[a, b, i], per (k, E', a, b, i).
    let mut left: [[[[[[[Field; 2]; BLOCK]; N3D]; NB]; NA]; NE]; NKZ] = [[[[[[[0; 2]; BLOCK]; N3D]; NB]; NA]; NE]; NKZ];
    for a in 0..NA {
        for b in 0..NB {
            let pos = index_pos::<INDEX_BITS>(neigh_idx[a][b], GA);
            for k in 0..NKZ {
                for e in 0..NE {
                    let gathered = G[k][e][pos];
                    for i in 0..N3D {
                        left[k][e][a][b][i] = complex_matmul(gathered, dH[a][b][i]);
                    }
                }
            }
        }
    }

    // dH[a, b, j] * D[q, w, a, b, i, j], per (q, w, a, b, i, j).
    let mut right: [[[[[[[[Field; 2]; BLOCK]; N3D]; N3D]; NB]; NA]; NW]; NQZ] =
        [[[[[[[[0; 2]; BLOCK]; N3D]; N3D]; NB]; NA]; NW]; NQZ];
    for qz in 0..NQZ {
        for w in 0..NW {
            for a in 0..NA {
                for b in 0..NB {
                    for i in 0..N3D {
                        for j in 0..N3D {
                            let f = D[qz][w][a][b][i][j];
                            for t in 0..BLOCK {
                                let h = dH[a][b][j][t];
                                right[qz][w][a][b][i][j][t] =
                                    [mul(h[0], f[0]) - mul(h[1], f[1]), mul(h[0], f[1]) + mul(h[1], f[0])];
                            }
                        }
                    }
                }
            }
        }
    }

    // Sigma[k, E, a] += left @ right.
    let mut sigma = Sigma;
    for k in 0..NKZ {
        for e in 0..NE {
            for qz in 0..NQZ {
                for w in 0..NW {
                    if w <= e {
                        for i in 0..N3D {
                            for j in 0..N3D {
                                for a in 0..NA {
                                    for b in 0..NB {
                                        let product = complex_matmul(left[k][e - w][a][b][i], right[qz][w][a][b][i][j]);
                                        for t in 0..BLOCK {
                                            sigma[k][e][a][t][0] += product[t][0];
                                            sigma[k][e][a][t][1] += product[t][1];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    sigma
}
