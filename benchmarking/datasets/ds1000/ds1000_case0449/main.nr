// DS-1000 case 0449 -- Noir baseline. Same as 0448.

fn main(x_dists: [Field; 12], y_dists: [Field; 12]) -> pub [Field; 24] {
    let mut out: [Field; 24] = [0; 24];
    for i in 0..3 {
        for j in 0..4 {
            out[(i * 4 + j) * 2] = x_dists[i * 4 + j];
            out[(i * 4 + j) * 2 + 1] = y_dists[i * 4 + j];
        }
    }
    out
}
