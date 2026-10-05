// DS-1000 case 0335 -- Noir baseline. result = np.max([a,b,c], axis=0) on Float[50].
// Q32 fixed-point. Per cell: max(max(a,b), c).

use noir_fpc::qgt;

fn main(a: [Field; 50], b: [Field; 50], c: [Field; 50]) -> pub [Field; 50] {
    let mut out: [Field; 50] = [0; 50];
    for k in 0..50 {
        let ab_max = if qgt(b[k], a[k]) { b[k] } else { a[k] };
        let final_max = if qgt(c[k], ab_max) { c[k] } else { ab_max };
        out[k] = final_max;
    }
    out
}
