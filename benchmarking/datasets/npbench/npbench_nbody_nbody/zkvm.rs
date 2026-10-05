//! NPBench nbody: a leapfrog N-body simulation with softened gravity,
//! recording the kinetic and potential energy after every step.
//!
//! ```python
//! def getAcc(pos, mass, G, softening):
//!     x = pos[:, 0:1]
//!     y = pos[:, 1:2]
//!     z = pos[:, 2:3]
//!     dx = x.T - x
//!     dy = y.T - y
//!     dz = z.T - z
//!     inv_r3 = dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2
//!     inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)
//!     ax = G * (dx * inv_r3) @ mass
//!     ay = G * (dy * inv_r3) @ mass
//!     az = G * (dz * inv_r3) @ mass
//!     a = np.hstack((ax, ay, az))
//!     return a
//!
//! def getEnergy(pos, vel, mass, G):
//!     KE = 0.5 * np.sum(mass * vel ** 2)
//!     x = pos[:, 0:1]
//!     y = pos[:, 1:2]
//!     z = pos[:, 2:3]
//!     dx = x.T - x
//!     dy = y.T - y
//!     dz = z.T - z
//!     inv_r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
//!     inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]
//!     PE = G * np.sum(np.triu(-(mass * mass.T) * inv_r, 1))
//!     return (KE, PE)
//!
//! @requires(lambda mass, pos, vel, N, Nt, dt, G, softening: 0 <= N <= 1024)
//! @requires(lambda mass, pos, vel, N, Nt, dt, G, softening: dt != 0.0)
//! def nbody(mass: Float[Array, '25 1'], pos: Float[Array, '25 3'], vel: Float[Array, '25 3'], N: Integer, Nt: Integer, dt: Float, G: Float, softening: Float):
//!     vel -= np.mean(mass * vel, axis=0) / np.mean(mass)
//!     acc = getAcc(pos, mass, G, softening)
//!     KE = np.ndarray(Nt + 1, dtype=np.float64)
//!     PE = np.ndarray(Nt + 1, dtype=np.float64)
//!     KE[0], PE[0] = getEnergy(pos, vel, mass, G)
//!     t = 0.0
//!     for i in range(Nt):
//!         vel += acc * dt / 2.0
//!         pos += vel * dt
//!         acc = getAcc(pos, mass, G, softening)
//!         vel += acc * dt / 2.0
//!         t += dt
//!         KE[i + 1], PE[i + 1] = getEnergy(pos, vel, mass, G)
//!     return (KE, PE)
//! ```
use crate::prelude::*;

/// `pos[j] - pos[i]` per coordinate: row `i`, column `j` of `x.T - x` etc.
fn displacement(pos: &Array2<Q32>, i: usize, j: usize) -> [Q32; 3] {
    core::array::from_fn(|c| pos[(j, c)] - pos[(i, c)])
}

/// `dx ** 2 + dy ** 2 + dz ** 2`.
fn squared_distance(d: [Q32; 3]) -> Q32 {
    d[0].powi(2) + d[1].powi(2) + d[2].powi(2)
}

/// `getAcc`: row `i` of `G * (d * inv_r3) @ mass` for each coordinate `d`,
/// one fused dot product over `j` per entry.
fn get_acc(pos: &Array2<Q32>, mass: ArrayView1<Q32>, g: Q32, softening: Q32) -> Array2<Q32> {
    let n = pos.nrows();
    let softening2 = softening.powi(2);
    let minus_three_halves = Q32::from_int(-3) / 2;
    let mut acc = Array2::zeros((n, 3));
    let mut weights = Array2::zeros((n, 3));
    for i in 0..n {
        // weights[j, c] = G * (d_c[i, j] * inv_r3[i, j])
        for (j, mut row) in weights.rows_mut().into_iter().enumerate() {
            let d = displacement(pos, i, j);
            let mut inv_r3 = squared_distance(d) + softening2;
            if inv_r3 > Q32::ZERO {
                inv_r3 = inv_r3.powf(minus_three_halves);
            }
            for (w, d) in row.iter_mut().zip(d) {
                *w = g * (d * inv_r3);
            }
        }
        acc.row_mut(i).assign(&np::vecmat(mass, weights.view()));
    }
    acc
}

/// `getEnergy`: the kinetic and the potential energy.
fn get_energy(pos: &Array2<Q32>, vel: &Array2<Q32>, mass: ArrayView1<Q32>, g: Q32) -> (Q32, Q32) {
    let half = Q32::ONE / 2;
    let kinetic: Q32 = Zip::from(vel.rows()).and(mass).fold(Q32::ZERO, |sum, v, &m| {
        sum + v.iter().map(|&v| m * v.powi(2)).sum::<Q32>()
    });
    // np.triu(..., 1) keeps the pairs i < j.
    let n = pos.nrows();
    let mut potential = Q32::ZERO;
    for i in 0..n {
        for j in i + 1..n {
            let mut inv_r = squared_distance(displacement(pos, i, j)).sqrt();
            if inv_r > Q32::ZERO {
                inv_r = Q32::ONE / inv_r;
            }
            potential += -(mass[i] * mass[j]) * inv_r;
        }
    }
    (half * kinetic, g * potential)
}

pub mod nbody {
    use super::*;

    /// `(KE, PE)`, `Nt + 1` entries each.
    pub type Output = (Array1<Q32>, Array1<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mass: Array2<Q32> = args.array2();
        let mut pos: Array2<Q32> = args.array2();
        let mut vel: Array2<Q32> = args.array2();
        let n_bodies: i64 = args.scalar();
        let nt: i64 = args.scalar();
        let dt: Q32 = args.scalar();
        let g: Q32 = args.scalar();
        let softening: Q32 = args.scalar();
        // N is read only by the precondition.
        assert!((0..=1024).contains(&n_bodies), "requires 0 <= N <= 1024");
        assert!(dt != Q32::ZERO, "requires dt != 0.0");
        assert_eq!(mass.ncols(), 1, "mass is a column of N masses");
        let mass = mass.column(0);
        let steps = usize::try_from(nt).expect("np.ndarray(Nt + 1): negative dimensions are not allowed");

        // vel -= np.mean(mass * vel, axis=0) / np.mean(mass)
        let count = mass.len() as i64;
        let mean_mass = mass.sum() / count;
        let drift: Array1<Q32> = vel
            .columns()
            .into_iter()
            .map(|v| Zip::from(v).and(mass).fold(Q32::ZERO, |sum, &v, &m| sum + m * v) / count / mean_mass)
            .collect();
        vel -= &drift;

        let mut acc = get_acc(&pos, mass, g, softening);
        let (mut ke, mut pe) = (Vec::with_capacity(steps + 1), Vec::with_capacity(steps + 1));
        let (k, p) = get_energy(&pos, &vel, mass, g);
        ke.push(k);
        pe.push(p);
        for _ in 0..steps {
            vel.zip_mut_with(&acc, |v, &a| *v += a * dt / 2);
            pos.zip_mut_with(&vel, |x, &v| *x += v * dt);
            acc = get_acc(&pos, mass, g, softening);
            vel.zip_mut_with(&acc, |v, &a| *v += a * dt / 2);
            let (k, p) = get_energy(&pos, &vel, mass, g);
            ke.push(k);
            pe.push(p);
        }
        (Array1::from(ke), Array1::from(pe))
    }
}
