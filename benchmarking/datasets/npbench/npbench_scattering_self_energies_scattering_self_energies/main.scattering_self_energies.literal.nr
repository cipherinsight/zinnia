// npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies: literal translation of kernel.py (LITERAL_BASELINES.md).
// All eight source loops and three statements retained; no factor/gather hoisting. Complex matmul uses current optimized arithmetic: four rounded real products per term, then running sums (fairness contract).
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

fn product(x: [Field; 2], y: [Field; 2]) -> [Field; 2] {
    [mul(x[0], y[0]) - mul(x[1], y[1]), mul(x[0], y[1]) + mul(x[1], y[0])]
}
fn matmul(x: [[Field; 2]; BLOCK], y: [[Field; 2]; BLOCK]) -> [[Field; 2]; BLOCK] {
    let mut result = [[0; 2]; BLOCK];
    for r in 0..NORB {
        for c in 0..NORB {
            let mut acc = [0, 0];
            for t in 0..NORB {
                let term = product(x[r * NORB + t], y[t * NORB + c]);
                acc = [acc[0] + term[0], acc[1] + term[1]];
            }
            result[r * NORB + c] = acc;
        }
    }
    result
}
fn main(
    neigh_idx: [[Field; NB]; NA],
    dH: [[[[[Field; 2]; BLOCK]; N3D]; NB]; NA],
    G: [[[[[Field; 2]; BLOCK]; GA]; NE]; NKZ],
    D: [[[[[[[Field; 2]; N3D]; N3D]; NB]; NA]; NW]; NQZ],
    Sigma: [[[[[Field; 2]; BLOCK]; NA]; NE]; NKZ],
) -> pub [[[[[Field; 2]; BLOCK]; NA]; NE]; NKZ] {
    // Input domain: every element of dH, G, D and Sigma is a Q32 code in the
    // window. neigh_idx is also checked once below.
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

    for a in 0..NA {
        check_inputs(neigh_idx[a]);
    }
    let mut sigma = Sigma;
    for k in 0..NKZ {
        for e in 0..NE {
            for qz in 0..NQZ {
                for w in 0..NW {
                    for i in 0..N3D {
                        for j in 0..N3D {
                            for a in 0..NA {
                                for b in 0..NB {
                                    if e >= w {
                                        let gathered = G[k][e - w][index_pos::<INDEX_BITS>(
                                            neigh_idx[a][b],
                                            GA,
                                        )];
                                        let dhg = matmul(gathered, dH[a][b][i]);
                                        let mut dhd = [[0; 2]; BLOCK];
                                        for t in 0..BLOCK {
                                            dhd[t] = product(dH[a][b][j][t], D[qz][w][a][b][i][j]);
                                        }
                                        let update = matmul(dhg, dhd);
                                        for t in 0..BLOCK {
                                            sigma[k][e][a][t] = [
                                                sigma[k][e][a][t][0] + update[t][0],
                                                sigma[k][e][a][t][1] + update[t][1],
                                            ];
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
