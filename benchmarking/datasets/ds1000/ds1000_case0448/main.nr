// DS-1000 case 0448, Noir baseline.
// Kernel: `np.vstack(([x_dists.T], [y_dists.T])).T` on two Float[3, 4]: the
// (3, 4, 2) result pairs x_dists[i, j] with y_dists[i, j].
// @outputs: float[24]

fn main(x_dists: [Field; 12], y_dists: [Field; 12]) -> pub [Field; 24] {
    let mut out = [0; 24];
    for k in 0..12 {
        out[2 * k] = x_dists[k];
        out[2 * k + 1] = y_dists[k];
    }
    out
}
