// DS-1000 case 0413 -- Noir baseline. x[x.imag != 0] on real Float[10].
// For real inputs, imag is identically 0, so the mask is empty. Sentinel zero output.

fn main(x: [Field; 10]) -> pub Field {
    let _ = x[0];
    0
}
