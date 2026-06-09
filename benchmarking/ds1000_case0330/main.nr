// DS-1000 case 0330 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// result = a ** power on Float[8,6].

use noir_fpc::{qpow_int_q32, qge};

fn main(a: [Field; 48], power: Field) -> pub [Field; 48] {
    let zero: Field = 0;
    let power_nonneg = qge(power, zero);
    let power_u64 = power as u64;
    let mut out: [Field; 48] = [0; 48];
    for k in 0..48 {
        let nonzero_base = a[k] != zero;
        assert(nonzero_base | power_nonneg);
        out[k] = qpow_int_q32(a[k], power_u64);
    }
    out
}
