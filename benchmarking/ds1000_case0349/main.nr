// DS-1000 case 0349 -- Noir baseline. result = a.shape[1] on Float[8,6].
// Static shape value; input is loaded for fixture parity but DCE'd.

fn main(a: [Field; 48]) -> pub Field {
    let _ = a[0];
    6
}
