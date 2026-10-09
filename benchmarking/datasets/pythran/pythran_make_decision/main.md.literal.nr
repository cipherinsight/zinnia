// pythran_make_decision__md: literal translation of kernel.py (LITERAL_BASELINES.md).
// Complex abs follows the current optimized Q32 schedule (rounded component squares, sqrt). Full arrays, first-index argmin scan, then standard gather.
// @outputs: float[2 * L]

use noir_hw::{check_inputs, lt, mul, sqrt};

// Sizes (hw_check.sizes): from fixture.json.
global L: u32 = 16; // = E_0
global M: u32 = 8; // = symbols_0

global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (symbols_0 - 1) // 128)

fn main(E: [[Field; 2]; L], symbols: [[Field; 2]; M]) -> pub [[Field; 2]; L] {
    for i in 0..L {
        check_inputs(E[i]);
    }
    for k in 0..M {
        check_inputs(symbols[k]);
    }
    let mut result = [[0; 2]; L];
    for i in 0..L {
        let mut difference = [[0; 2]; M];
        for k in 0..M {
            difference[k] = [E[i][0] - symbols[k][0], E[i][1] - symbols[k][1]];
        }
        let mut magnitude = [0; M];
        for k in 0..M {
            magnitude[k] = sqrt(
                mul(difference[k][0], difference[k][0]) + mul(difference[k][1], difference[k][1]),
            );
        }
        let mut distances = [0; M];
        for k in 0..M {
            distances[k] = mul(magnitude[k], magnitude[k]);
        }
        let mut best = distances[0];
        let mut im: Field = 0;
        for k in 1..M {
            let better = lt(distances[k], best);
            best = if better { distances[k] } else { best };
            im = if better { k as Field } else { im };
        }
        result[i] = symbols[noir_hw::index_pos::<INDEX_BITS>(im, M)];
    }
    result
}
