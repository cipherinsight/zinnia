// DS-1000 case 0314 -- Noir baseline.
// Replace first argmax with a.min(), then return unravel of second argmax.

use noir_fpc::{qgt, qlt};

global COLS: u32 = 6;

fn main(a: [Field; 48]) -> pub [Field; 2] {
    let mut buf: [Field; 48] = a;
    let mut amax: u32 = 0;
    let mut vmax: Field = buf[0];
    let mut vmin: Field = buf[0];
    for k in 1..48 {
        if qgt(buf[k], vmax) { vmax = buf[k]; amax = k; }
        if qlt(buf[k], vmin) { vmin = buf[k]; }
    }
    buf[amax] = vmin;
    let mut amax2: u32 = 0;
    let mut vmax2: Field = buf[0];
    for k in 1..48 {
        if qgt(buf[k], vmax2) { vmax2 = buf[k]; amax2 = k; }
    }
    [(amax2 / COLS) as Field, (amax2 % COLS) as Field]
}
