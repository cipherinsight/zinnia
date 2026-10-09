// pythran_periodic_dist__dist: literal translation of kernel.py (LITERAL_BASELINES.md).
// Tile/transpose are wires. Mask reads use L5 equality-test compaction; masked writes scatter by count equality. Private flags select complete updated arrays; no symmetry.
// @outputs: float[PAIRS]; float[PAIRS]; float[PAIRS]; float[PAIRS]
global N: u32 = 16; // = x_0
global PAIRS: u32 = 256; // = x_0 * x_0
// RHS compaction and separate LHS mask; all slots retained internally with a private count.
fn update(values: [Field; PAIRS], L: Field, negative: bool) -> [Field; PAIRS] {
    let signed_l = if negative { -L } else { L };
    let half = noir_hw::div(noir_hw::from_int(signed_l), 8589934592);
    let mut keep = [false; PAIRS];
    for i in 0..PAIRS {
        keep[i] = if negative {
            noir_hw::lt(values[i], half)
        } else {
            noir_hw::gt(values[i], half)
        };
    }
    let mut count: Field = 0;
    let mut positions = [0; PAIRS];
    for i in 0..PAIRS {
        positions[i] = count;
        count += keep[i] as Field;
    }
    let mut compacted = [0; PAIRS];
    for j in 0..PAIRS {
        let mut sum: Field = 0;
        for i in 0..PAIRS {
            sum += ((keep[i] & (positions[i] == j as Field)) as Field) * values[i];
        }
        compacted[j] = sum;
    }
    let lf = noir_hw::from_int(L);
    let mut changed = [0; PAIRS];
    for j in 0..PAIRS {
        changed[j] = if negative {
            compacted[j] + lf
        } else {
            compacted[j] - lf
        };
    }
    let signed_l = if negative { -L } else { L };
    let half = noir_hw::div(noir_hw::from_int(signed_l), 8589934592);
    let mut target_keep = [false; PAIRS];
    for i in 0..PAIRS {
        target_keep[i] = if negative {
            noir_hw::lt(values[i], half)
        } else {
            noir_hw::gt(values[i], half)
        };
    }
    let mut next: Field = 0;
    let mut target_positions = [0; PAIRS];
    for i in 0..PAIRS {
        target_positions[i] = next;
        next += target_keep[i] as Field;
    }
    let mut result = values;
    for i in 0..PAIRS {
        let mut replacement: Field = 0;
        for j in 0..PAIRS {
            replacement += ((target_positions[i] == j as Field) as Field) * changed[j];
        }
        result[i] = if target_keep[i] {
            replacement
        } else {
            values[i]
        };
    }
    result
}

fn main(
    x: [Field; N],
    y: [Field; N],
    z: [Field; N],
    L: Field,
    periodicX: bool,
    periodicY: bool,
    periodicZ: bool,
) -> pub ([Field; PAIRS], [Field; PAIRS], [Field; PAIRS], [Field; PAIRS]) {
    noir_hw::check_inputs(x);
    noir_hw::check_inputs(y);
    noir_hw::check_inputs(z);
    noir_hw::check_input(L);
    let mut xtemp = [0; PAIRS];
    for i in 0..N {
        for j in 0..N {
            xtemp[i * N + j] = x[j];
        }
    }
    let mut dx = [0; PAIRS];
    for i in 0..N {
        for j in 0..N {
            dx[i * N + j] = xtemp[i * N + j] - xtemp[j * N + i];
        }
    }
    let mut ytemp = [0; PAIRS];
    for i in 0..N {
        for j in 0..N {
            ytemp[i * N + j] = y[j];
        }
    }
    let mut dy = [0; PAIRS];
    for i in 0..N {
        for j in 0..N {
            dy[i * N + j] = ytemp[i * N + j] - ytemp[j * N + i];
        }
    }
    let mut ztemp = [0; PAIRS];
    for i in 0..N {
        for j in 0..N {
            ztemp[i * N + j] = z[j];
        }
    }
    let mut dz = [0; PAIRS];
    for i in 0..N {
        for j in 0..N {
            dz[i * N + j] = ztemp[i * N + j] - ztemp[j * N + i];
        }
    }
    let first = update(dx, L, false);
    let second = update(first, L, true);
    for at in 0..PAIRS {
        dx[at] = if periodicX { second[at] } else { dx[at] };
    }
    let first = update(dy, L, false);
    let second = update(first, L, true);
    for at in 0..PAIRS {
        dy[at] = if periodicY { second[at] } else { dy[at] };
    }
    let first = update(dz, L, false);
    let second = update(first, L, true);
    for at in 0..PAIRS {
        dz[at] = if periodicZ { second[at] } else { dz[at] };
    }
    let mut sx = [0; PAIRS];
    for at in 0..PAIRS {
        sx[at] = noir_hw::mul(dx[at], dx[at]);
    }
    let mut sy = [0; PAIRS];
    for at in 0..PAIRS {
        sy[at] = noir_hw::mul(dy[at], dy[at]);
    }
    let mut xy = [0; PAIRS];
    for at in 0..PAIRS {
        xy[at] = sx[at] + sy[at];
    }
    let mut sz = [0; PAIRS];
    for at in 0..PAIRS {
        sz[at] = noir_hw::mul(dz[at], dz[at]);
    }
    let mut xyz = [0; PAIRS];
    for at in 0..PAIRS {
        xyz[at] = xy[at] + sz[at];
    }
    let mut roots = [0; PAIRS];
    for at in 0..PAIRS {
        roots[at] = noir_hw::sqrt(xyz[at]);
    }
    let mut mask = [false; PAIRS];
    for at in 0..PAIRS {
        mask[at] = roots[at] == 0;
    }
    let mut d = roots;
    for at in 0..PAIRS {
        d[at] = if mask[at] { -4294967296 } else { roots[at] };
    }
    (d, dx, dy, dz)
}
