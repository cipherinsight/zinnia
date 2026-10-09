// pythran_spectraldns__add_pressure_diffusion: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize full complex arrays per expression, with no CSE or dead-write elimination.
// Real-complex multiplication is two Q32 products; multiplication by 1j uses the complex definition.

// @outputs: float[2 * POINTS]

use noir_hw::{check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.add_pressure_diffusion.json.
global A1: u32 = 2; // = dU_1
global A2: u32 = 5; // = dU_2
global A3: u32 = 7; // = dU_3
global POINTS: u32 = A1 * A2 * A3;

fn main(
    dU: [[[Field; 2]; POINTS]; 3],
    U_hat: [[[Field; 2]; POINTS]; 3],
    K2: [Field; POINTS],
    K: [[Field; POINTS]; 3],
    K_over_K2: [[Field; POINTS]; 3],
    nu: Field,
) -> pub [[Field; 2]; POINTS] {
    noir_hw::check_input(nu);
    check_inputs(K2);
    for k in 0..3 {
        check_inputs(K[k]);
        check_inputs(K_over_K2[k]);
        for p in 0..POINTS {
            check_inputs(dU[k][p]);
            check_inputs(U_hat[k][p]);
        }
    }
    let mut first = [[0; 2]; POINTS];
    for p in 0..POINTS {
        for part in 0..2 {
            first[p][part] = mul(dU[0][p][part], K_over_K2[0][p]);
        }
    }
    let mut second = [[0; 2]; POINTS];
    for p in 0..POINTS {
        for part in 0..2 {
            second[p][part] = mul(dU[1][p][part], K_over_K2[1][p]);
        }
    }
    let mut sum = [[0; 2]; POINTS];
    for p in 0..POINTS {
        for part in 0..2 {
            sum[p][part] = first[p][part] + second[p][part];
        }
    }
    let mut third = [[0; 2]; POINTS];
    for p in 0..POINTS {
        for part in 0..2 {
            third[p][part] = mul(dU[2][p][part], K_over_K2[2][p]);
        }
    }
    let mut ph = [[0; 2]; POINTS];
    for p in 0..POINTS {
        for part in 0..2 {
            ph[p][part] = sum[p][part] + third[p][part];
        }
    }
    let mut pressure = [[[0; 2]; POINTS]; 3];
    for k in 0..3 {
        for p in 0..POINTS {
            for part in 0..2 {
                pressure[k][p][part] = mul(ph[p][part], K[k][p]);
            }
        }
    }
    let mut difference = [[[0; 2]; POINTS]; 3];
    for k in 0..3 {
        for p in 0..POINTS {
            for part in 0..2 {
                difference[k][p][part] = dU[k][p][part] - pressure[k][p][part];
            }
        }
    }
    let mut viscosity = [0; POINTS];
    for p in 0..POINTS {
        viscosity[p] = mul(nu, K2[p]);
    }
    let mut diffusion = [[[0; 2]; POINTS]; 3];
    for k in 0..3 {
        for p in 0..POINTS {
            for part in 0..2 {
                diffusion[k][p][part] = mul(viscosity[p], U_hat[k][p][part]);
            }
        }
    }
    let mut du = dU;
    for k in 0..3 {
        for p in 0..POINTS {
            for part in 0..2 {
                du[k][p][part] = difference[k][p][part] - diffusion[k][p][part];
            }
        }
    }
    ph
}
