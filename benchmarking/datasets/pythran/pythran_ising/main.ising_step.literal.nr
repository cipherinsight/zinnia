// pythran_ising__ising_step: literal translation of kernel.py (LITERAL_BASELINES.md).
// Evaluate both branches and select per update; float32 remains Q32; clamp and update mirror the source helpers.
// @outputs: int[N * M]

use noir_hw::{check_input, check_inputs, canonical_exp, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 2; // = field_0
global M: u32 = 2; // = field_1

// Input is v+1 so constant neighbour indices stay unsigned.
fn clamp(v: u32,n: u32) -> u32 {
    if v == 0 { n-1 } else if v-1 < n { v-1 } else { v-1-n }
}
fn update(mut f: [[Field; M]; N], n: u32,m: u32,beta: Field) -> [[Field; M]; N] {
    let mut total: Field = 0;
    for i in 0..3 { for j in 0..3 {
        if (i != 1) | (j != 1) { total += f[clamp(n+i,N)][clamp(m+j,M)]; }
    } }
    let de=2*f[n][m]*total;
    let first=noir_hw::le(de,0);
    let first_value=f[n][m]*(-1);
    let exponential=canonical_exp((-de)*beta);
    let second=noir_hw::gt(exponential,SCALE/2);
    let second_value=f[n][m]*(-1);
    let otherwise=if second { second_value } else { f[n][m] };
    f[n][m]=if first { first_value } else { otherwise };
    f
}

fn main(field: [[Field; M]; N], beta: Field) -> pub [[Field; M]; N] {
    // Input domain: field holds int64s, beta a Q32 code, in the window.
    for n in 0..N {
        check_inputs(field[n]);
    }
    check_input(beta);
    let mut f = field;
    for n_offset in 0..2 {
        for m_offset in 0..2 {
            for n in n_offset..N {
                for m in m_offset..M {
                    if ((n - n_offset) % 2 == 0) & ((m - m_offset) % 2 == 0) {
                        f=update(f,n,m,beta);
                    }
                }
            }
        }
    }
    f
}
