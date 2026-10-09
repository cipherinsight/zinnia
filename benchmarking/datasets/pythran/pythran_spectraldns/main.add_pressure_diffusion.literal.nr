// pythran_spectraldns__add_pressure_diffusion: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize full complex arrays per expression, with no CSE or dead-write elimination.
// Input checks match optimized P_hat dependence; every discarded dU product remains.
// Discarded products preserve Q32 floors through small shared divisions and exact
// field reconstruction, without windows absent from optimized. Low limbs carry
// rounding through the second product; standard constrained bit decomposition only.

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
    for k in 0..3 {
        check_inputs(K_over_K2[k]);
        for p in 0..POINTS {
            check_inputs(dU[k][p]);
        }
    }
    let mut uh_low = [[[0; 2]; POINTS]; 3];
    let mut k_low = [[0; POINTS]; 3];
    let mut k2_low = [[0; 2]; POINTS];
    for component in 0..3 {
        for p in 0..POINTS {
            k_low[component][p] = discarded_limbs(K[component][p])[0];
            for part in 0..2 {
                uh_low[component][p][part] = discarded_limbs(U_hat[component][p][part])[0];
            }
        }
    }
    for p in 0..POINTS { k2_low[p] = discarded_limbs(K2[p]); }
    let nu_low = discarded_limbs(nu);
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
                let ph_low = noir_hw::divmod_pos(ph[p][part], noir_hw::SCALE).1;
                pressure[k][p][part] = discarded_product(ph[p][part], K[k][p], ph_low, k_low[k][p]).0;
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
    let mut viscosity_low = [0; POINTS];
    for p in 0..POINTS {
let (value, low_quotient) = discarded_product(nu, K2[p], nu_low[0], k2_low[p][0]);
        viscosity[p] = value;
        let cross0 = nu_low[0] * k2_low[p][1];
        let cross1 = nu_low[1] * k2_low[p][0];
        viscosity_low[p] = noir_hw::divmod_pos(low_quotient + cross0 + cross1, noir_hw::SCALE).1;
    }
    let mut diffusion = [[[0; 2]; POINTS]; 3];
    for k in 0..3 {
        for p in 0..POINTS {
            for part in 0..2 {
                diffusion[k][p][part] = discarded_product(viscosity[p], U_hat[k][p][part], viscosity_low[p], uh_low[k][p][part]).0;
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

// Input representation only. Signed low limbs are congruent to the actual signed
// input modulo SCALE^2, including inputs outside the normal signed64 window.
global FIELD_MAG_BITS: u32 = 253;
global LIMB_BITS: u32 = 32;
global HALF_MODULUS: Field = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
fn discarded_limbs(x: Field) -> [Field; 2] {
    let negative = !x.lt(HALF_MODULUS + 1);
    let magnitude = if negative { -x } else { x };
    let bits: [bool; FIELD_MAG_BITS] = magnitude.to_le_bits();
    let mut limbs = [0; 2];
    for limb in 0..2 {
        let mut weight: Field = 1;
        for bit in 0..LIMB_BITS {
            limbs[limb] += (bits[limb * LIMB_BITS + bit] as Field) * weight;
            weight *= 2;
        }
        limbs[limb] = if negative { -limbs[limb] } else { limbs[limb] };
    }
    limbs
}

// The proven small remainder makes this exact field division the Q32 floor,
// even when the full discarded result is wider than signed64 or the field.
fn discarded_product(a: Field, b: Field, a_low: Field, b_low: Field) -> (Field, Field) {
    let raw = a * b;
    let low_raw = a_low * b_low;
    let (low_quotient, remainder) = noir_hw::divmod_pos(low_raw, noir_hw::SCALE);
    ((raw - remainder) / noir_hw::SCALE, low_quotient)
}
