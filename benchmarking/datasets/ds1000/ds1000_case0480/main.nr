// DS-1000 case 0480 -- Noir baseline (semantic translation).
// Python:
//   result = ((x == a) & (y == b)).argmax()
//   if x[result] != a or y[result] != b:
//       result = -1
//   return result
//
// Step-by-step Noir translation:
//   1. Build the boolean mask array `mask[k] = (x[k] == a) & (y[k] == b_val)`.
//   2. argmax over the mask: first index where the maximum is attained
//      (numpy's argmax returns the FIRST max; for an all-zero mask that's 0).
//   3. Post-check x[result] != a or y[result] != b_val; if so, return -1.
//
// Dynamic gather x[result] / y[result] is done with a linear scan
// (Noir doesn't support dynamic array indexing on `&[Field; N]` in a
// constraint-friendly way without it).

global N: u32 = 8;

fn main(x: [Field; 8], y: [Field; 8], a: Field, b_val: Field) -> pub Field {
    // Step 1: build the mask array.
    let mut mask: [u32; 8] = [0; 8];
    for k in 0..N {
        let m = (x[k] == a) & (y[k] == b_val);
        mask[k] = if m { 1 } else { 0 };
    }

    // Step 2: argmax. First index of the maximum value.
    let mut best_val: u32 = mask[0];
    let mut best_idx: u32 = 0;
    for k in 1..N {
        if mask[k] > best_val {
            best_val = mask[k];
            best_idx = k;
        }
    }

    // Step 3: gather x[best_idx], y[best_idx] (dynamic select).
    let mut xr: Field = 0;
    let mut yr: Field = 0;
    for k in 0..N {
        if k == best_idx {
            xr = x[k];
            yr = y[k];
        }
    }

    // Final: if x[result] != a or y[result] != b_val, return -1.
    if (xr != a) | (yr != b_val) { -1 } else { best_idx as Field }
}
